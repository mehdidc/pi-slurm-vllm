#!/usr/bin/env python3
"""OpenAI-compatible proxy that starts/reuses a Slurm vLLM job for Pi."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import http.client
import json
import os
import socket
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit


HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def content_to_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                item_type = item.get("type")
                if item_type in {"text", "input_text"} and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
        return "\n".join(part for part in parts if part)
    return str(content)


def normalize_chat_messages(payload: dict[str, object]) -> bool:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return False

    system_parts: list[str] = []
    normalized: list[object] = []
    changed = False
    for message in messages:
        if not isinstance(message, dict):
            normalized.append(message)
            continue
        role = message.get("role")
        if role in {"system", "developer"}:
            text = content_to_text(message.get("content"))
            if text:
                label = "Developer" if role == "developer" else "System"
                system_parts.append(f"{label} instructions:\n{text}")
            changed = True
            continue
        if role == "function":
            updated = dict(message)
            updated["role"] = "tool"
            normalized.append(updated)
            changed = True
            continue
        normalized.append(message)

    if system_parts:
        normalized.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
    if changed:
        payload["messages"] = normalized
    return changed


class State:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.backend_base_url: Optional[str] = None
        self.tunnel_process: Optional[subprocess.Popen[bytes]] = None
        self.last_event = "proxy initialized"
        self.last_error: Optional[str] = None
        self.job_id: Optional[str] = None
        self.job_state: Optional[str] = None
        self.job_nodes: Optional[str] = None
        self.request_count = 0


def run_text(cmd: list[str], timeout: int = 30, check: bool = True) -> str:
    proc = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"{' '.join(cmd)} failed with {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout.strip()


def health_ok(base_url: str, timeout: float = 2.0) -> bool:
    try:
        parsed = urlsplit(base_url)
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
        conn.request("GET", "/v1/models")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        return 200 <= resp.status < 500
    except OSError:
        return False


def find_existing_job(job_name: str, user: str) -> Optional[tuple[str, str, str]]:
    out = run_text(
        ["squeue", "-u", user, "-h", "-n", job_name, "-o", "%i|%T|%N"],
        check=False,
    )
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        job_id, state, nodes = parts
        if state in {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING"}:
            return job_id, state, nodes
    return None


def get_job_state(job_id: Optional[str]) -> Optional[tuple[str, str]]:
    if not job_id:
        return None
    out = run_text(["squeue", "-j", job_id, "-h", "-o", "%T|%N"], check=False)
    if not out:
        return None
    state, nodes = out.splitlines()[0].split("|", 1)
    return state, nodes


def cached_backend_ok(state: State) -> bool:
    if not state.backend_base_url or not health_ok(state.backend_base_url):
        return False
    job = get_job_state(state.job_id)
    if not job:
        record(state, f"Cached backend job {state.job_id} is no longer in squeue")
        state.backend_base_url = None
        state.job_state = None
        return False
    job_state, nodes = job
    state.job_state = job_state
    state.job_nodes = nodes
    if job_state != "RUNNING":
        record(state, f"Cached backend job {state.job_id} is {job_state}")
        state.backend_base_url = None
        return False
    return True


def submit_job(sbatch_path: str, env: dict[str, str]) -> str:
    out = subprocess.run(
        ["sbatch", "--parsable", sbatch_path],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    if out.returncode != 0:
        raise RuntimeError(f"sbatch failed\nstdout:\n{out.stdout}\nstderr:\n{out.stderr}")
    return out.stdout.strip().split(";")[0]


def wait_for_running(job_id: str, timeout_s: int, poll_s: int) -> tuple[str, str]:
    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        out = run_text(["squeue", "-j", job_id, "-h", "-o", "%T|%N"], check=False)
        if out:
            last = out.splitlines()[0]
            state, nodes = last.split("|", 1)
            if state == "RUNNING" and nodes and nodes != "(None)":
                return state, nodes
            if state in {"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}:
                raise RuntimeError(f"Slurm job {job_id} ended early: {state}")
        time.sleep(poll_s)
    raise TimeoutError(f"Timed out waiting for Slurm job {job_id}; last state: {last}")


def first_hostname(nodelist: str) -> str:
    out = run_text(["scontrol", "show", "hostnames", nodelist])
    first = out.splitlines()[0].strip()
    if not first:
        raise RuntimeError(f"Could not resolve Slurm nodelist {nodelist!r}")
    return first


def resolve_host(host: str) -> str:
    return socket.gethostbyname(host)


def wait_for_backend(base_url: str, timeout_s: int, poll_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if health_ok(base_url, timeout=5):
            return
        time.sleep(poll_s)
    raise TimeoutError(f"Timed out waiting for vLLM backend at {base_url}")


def ensure_tunnel(state: State, remote_host: str) -> str:
    args = state.args
    if not args.ssh_tunnel:
        return f"http://{remote_host}:{args.vllm_port}"

    local_url = f"http://127.0.0.1:{args.backend_local_port}"
    if health_ok(local_url):
        return local_url

    if state.tunnel_process and state.tunnel_process.poll() is None:
        state.tunnel_process.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            state.tunnel_process.wait(timeout=5)

    destination = args.ssh_host or remote_host
    cmd = [
        "ssh",
        "-N",
        "-L",
        f"127.0.0.1:{args.backend_local_port}:{remote_host}:{args.vllm_port}",
        destination,
    ]
    state.tunnel_process = subprocess.Popen(cmd)
    return local_url


def record(state: State, event: str) -> None:
    state.last_event = event
    print(event, flush=True)


def ensure_backend(state: State) -> str:
    args = state.args
    if cached_backend_ok(state):
        return state.backend_base_url

    lock_path = Path(args.lock_file)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cached_backend_ok(state):
            return state.backend_base_url

        user = args.slurm_user or os.environ.get("USER") or run_text(["whoami"])
        existing = find_existing_job(args.job_name, user)
        if existing:
            job_id, job_state, nodes = existing
            state.job_id = job_id
            state.job_state = job_state
            state.job_nodes = nodes
            record(state, f"Using existing Slurm job {job_id} ({job_state})")
        else:
            env = os.environ.copy()
            env.setdefault("PI_VLLM_PORT", str(args.vllm_port))
            env.setdefault("PI_VLLM_SERVED_MODEL_NAME", args.model_id)
            record(state, f"Submitting Slurm job with {args.sbatch}")
            job_id = submit_job(args.sbatch, env)
            state.job_id = job_id
            state.job_state = "SUBMITTED"
            record(state, f"Submitted Slurm job {job_id}")

        record(state, f"Waiting for Slurm job {job_id} to start")
        _, nodes = wait_for_running(job_id, args.slurm_start_timeout, args.poll_interval)
        state.job_state = "RUNNING"
        state.job_nodes = nodes
        head_node = first_hostname(nodes)
        remote_host = resolve_host(head_node) if args.resolve_node_ip else head_node
        backend = ensure_tunnel(state, remote_host)
        wait_for_backend(backend, args.vllm_ready_timeout, args.poll_interval)
        state.backend_base_url = backend
        record(state, f"vLLM backend ready at {backend}")
        return backend


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        pass  # silence per-request logs

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self.send_json({"ok": True})
            return
        if self.path == "/status":
            state: State = self.server.state  # type: ignore[attr-defined]
            self.send_json({
                "ok": True,
                "request_count": state.request_count,
                "last_event": state.last_event,
                "last_error": state.last_error,
                "job_id": state.job_id,
                "job_state": state.job_state,
                "job_nodes": state.job_nodes,
                "backend_base_url": state.backend_base_url,
            })
            return
        self.forward()

    def send_json(self, data: dict[str, object], status: int = 200) -> None:
        payload = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        self.forward()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "authorization,content-type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def forward(self) -> None:
        state: State = self.server.state  # type: ignore[attr-defined]
        try:
            state.request_count += 1
            backend = ensure_backend(state)
            parsed = urlsplit(backend)
            body_len = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(body_len) if body_len else None

            path = self.path
            headers = {
                k: v
                for k, v in self.headers.items()
                if k.lower() not in HOP_BY_HOP_HEADERS and k.lower() != "host"
            }
            if (
                body
                and self.command == "POST"
                and path.split("?", 1)[0] == "/v1/chat/completions"
                and "json" in self.headers.get("Content-Type", "").lower()
            ):
                try:
                    payload = json.loads(body.decode("utf-8"))
                    if isinstance(payload, dict) and normalize_chat_messages(payload):
                        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                        headers["Content-Type"] = "application/json"
                except (UnicodeDecodeError, json.JSONDecodeError):
                    pass

            conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=state.args.request_timeout)
            headers["Host"] = parsed.netloc
            conn.request(self.command, path, body=body, headers=headers)
            resp = conn.getresponse()

            self.send_response(resp.status, resp.reason)
            for key, value in resp.getheaders():
                if key.lower() not in HOP_BY_HOP_HEADERS:
                    self.send_header(key, value)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Connection", "close")
            self.end_headers()

            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
            conn.close()
        except Exception as exc:
            state.last_error = str(exc)
            record(state, f"ERROR: {exc}")
            payload = json.dumps({"error": str(exc)}).encode()
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=8123)
    parser.add_argument("--job-name", default="pi-vllm-qwen36-27b")
    parser.add_argument("--sbatch", default="slurm/pi-vllm.sbatch")
    parser.add_argument("--model-id", default="Qwen3.6-27B-FP8")
    parser.add_argument("--vllm-port", type=int, default=8080)
    parser.add_argument("--slurm-user")
    parser.add_argument("--slurm-start-timeout", type=int, default=1800)
    parser.add_argument("--vllm-ready-timeout", type=int, default=1800)
    parser.add_argument("--poll-interval", type=int, default=10)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--lock-file", default="/tmp/pi-vllm-proxy.lock")
    parser.add_argument("--resolve-node-ip", action="store_true", default=True)
    parser.add_argument("--no-resolve-node-ip", dest="resolve_node_ip", action="store_false")
    parser.add_argument("--ssh-tunnel", action="store_true")
    parser.add_argument("--ssh-host", help="SSH target for tunnel. Defaults to the compute node.")
    parser.add_argument("--backend-local-port", type=int, default=18123)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.sbatch = str(Path(args.sbatch).resolve())
    state = State(args)
    server = ThreadingHTTPServer((args.listen_host, args.listen_port), ProxyHandler)
    server.state = state  # type: ignore[attr-defined]
    print(f"Pi vLLM proxy listening on http://{args.listen_host}:{args.listen_port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if state.tunnel_process and state.tunnel_process.poll() is None:
            state.tunnel_process.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
