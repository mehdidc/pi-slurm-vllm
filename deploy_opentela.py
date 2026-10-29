#!/usr/bin/env python3
"""Expose a repository Slurm model through OpenTela and print its API base URL."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import getpass
import hashlib
import json
import os
import platform
import re
import select
import shutil
import signal
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener, urlopen

import vllm_proxy as slurm


OTELA_VERSION = "v0.2.6"
# Digests from https://api.github.com/repos/eth-easl/OpenTela/releases/tags/v0.2.6
OTELA_SHA256 = {
    "amd64": "e437012ee04749434072ccfa7d1732dafb5a7fd5c4bf66780f52218cd282845c",
    "arm64": "0638fa17dc1e11685f0d13684916ce0998030fc2001910ec0a766cb7a809423e",
}
BOOTSTRAP = (
    # Public WSS bootstrap used by teraflop-ai/model-serving's HPC relay.
    "/dns4/p2p.opentela.ai/tcp/443/wss/p2p/"
    "QmTtnXKHvovCwkBZRR4NcxeHfnt5EJQgN4wo9KV8U8nYP7"
)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward a bearer credential to a redirected host.


def get_json(url: str, api_key: str = "", timeout: float = 10, *, use_proxy: bool = True):
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    handlers = [NoRedirect()]
    if not use_proxy:
        handlers.append(ProxyHandler({}))
    with build_opener(*handlers).open(Request(url, headers=headers), timeout=timeout) as response:
        return json.load(response)


def has_model(body, model: str) -> bool:
    return isinstance(body, dict) and isinstance(body.get("data"), list) and any(
        isinstance(item, dict) and item.get("id") == model for item in body["data"]
    )


def is_local_url(url: str) -> bool:
    return urlsplit(url).hostname in {"localhost", "127.0.0.1", "::1"}


def detect_cluster() -> str:
    available = {p.name for p in slurm.SUPPORTED_RUNNERS_ROOT.iterdir() if p.is_dir()}
    candidate = os.environ.get("SLURM_CLUSTER_NAME", "").lower()
    if candidate in available:
        return candidate
    try:
        config = slurm.run_text(["scontrol", "show", "config"], timeout=5)
        match = re.search(r"^\s*ClusterName\s*=\s*(\S+)", config, re.MULTILINE)
        if match and match[1].lower() in available:
            return match[1].lower()
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        pass
    host = socket.gethostname().lower()
    for cluster, prefix in (("jureca", "jr"), ("jupiter", "jp"), ("juwels", "jw")):
        if cluster in available and (cluster in host or host.startswith(prefix)):
            return cluster
    if len(available) == 1:
        return available.pop()
    raise RuntimeError(f"Cannot detect Slurm cluster; pass --cluster ({', '.join(sorted(available))}).")


def ensure_otela(explicit: str | None) -> str:
    if explicit:
        binary = shutil.which(explicit)
        if not binary:
            raise RuntimeError(f"OpenTela executable not found: {explicit}")
        return str(Path(binary).resolve())
    installed = shutil.which("otela") or shutil.which("opentela")
    if installed:
        return str(Path(installed).resolve())
    arch = {"x86_64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(platform.machine())
    if platform.system() != "Linux" or not arch:
        raise RuntimeError("Automatic OpenTela installation requires Linux amd64/arm64; use --otela PATH.")
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "pi-slurm-vllm"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"opentela-{OTELA_VERSION}-{arch}"
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == OTELA_SHA256[arch]:
        return str(target.resolve())
    url = f"https://github.com/eth-easl/OpenTela/releases/download/{OTELA_VERSION}/opentela-{arch}"
    log(f"Downloading OpenTela {OTELA_VERSION} ({arch})")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache, delete=False) as output:
            temporary = Path(output.name)
            with urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, output)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != OTELA_SHA256[arch]:
            raise RuntimeError("OpenTela download failed SHA-256 verification.")
        temporary.chmod(0o700)
        temporary.replace(target)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
    return str(target.resolve())


def job_state(job_id: str) -> tuple[str, str]:
    result = slurm.run_text(["squeue", "-j", job_id, "-h", "-o", "%T|%N"])
    if not result:
        raise RuntimeError(f"Slurm job {job_id} is no longer in the queue.")
    state, nodes = result.splitlines()[0].split("|", 1)
    if state not in {"RUNNING", "PENDING", "CONFIGURING"}:
        raise RuntimeError(f"Slurm job {job_id} is {state}.")
    return state, nodes


def backend_hosts(node: str, cluster: str, override: str | None = None) -> list[str]:
    if override:
        return [override.replace("{node}", node)]
    # JURECA/JUWELS runners bind vLLM to MASTER_ADDR="$head_node"i.
    # Slurm's node name resolves to a different network interface.
    if cluster in {"jureca", "juwels"}:
        short, dot, domain = node.partition(".")
        return [f"{short}i{dot}{domain}", node]
    return [node]


def start_backend(args, runner: Path) -> tuple[str, str, int]:
    # Share the proxy's submission lock so Pi and this script cannot both submit.
    with open("/tmp/vllm-proxy.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        queued = slurm.run_text([
            "squeue", "-u", getpass.getuser(), "-h", "-n", slurm.sbatch_job_name(runner),
            "-o", "%i|%T|%N",
        ])
        existing = None
        for line in queued.splitlines():
            fields = line.split("|", 2)
            if len(fields) == 3 and fields[1] in {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING"}:
                existing = fields
                break
        job_id = existing[0] if existing else slurm.submit_job(str(runner))
    log(f"{'Reusing' if existing else 'Submitted'} Slurm job {job_id}. "
        f"It is retained on exit; cancel explicitly with: scancel {job_id}")
    deadline = time.monotonic() + args.slurm_start_timeout
    while True:
        state, nodes = job_state(job_id)
        if state == "RUNNING" and nodes and nodes != "(None)":
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for Slurm job {job_id} to start.")
        time.sleep(args.poll_interval)
    node = slurm.first_hostname(nodes)
    hosts = backend_hosts(node, args.cluster, args.backend_host)
    port = slurm.sbatch_vllm_port(runner)
    model = slurm.sbatch_served_model_name(runner)
    log(f"Waiting for {model} at {', '.join(f'{host}:{port}' for host in hosts)}")
    deadline = time.monotonic() + args.vllm_ready_timeout
    next_report = 0.0
    failures = []
    while time.monotonic() < deadline:
        job_state(job_id)
        failures = []
        for host in hosts:
            try:
                body = get_json(f"http://{host}:{port}/v1/models", timeout=2, use_proxy=False)
                if has_model(body, model):
                    log(f"vLLM ready at {host}:{port}")
                    return job_id, host, port
                reason = f"response does not list model {model}"
            except (OSError, ValueError, URLError) as exc:
                reason = str(exc)
            failures.append(f"{host}:{port}: {reason}")
        if time.monotonic() >= next_report:
            log("Still waiting for vLLM: " + "; ".join(failures))
            next_report = time.monotonic() + 30
        time.sleep(args.poll_interval)
    raise TimeoutError(f"Timed out waiting for model {model} on Slurm job {job_id}: "
                       + "; ".join(failures))


class BridgeHandler(socketserver.BaseRequestHandler):
    """Forward bytes unchanged, including streaming responses and tool calls."""

    def handle(self):
        try:
            with socket.create_connection(self.server.backend, timeout=10) as backend:
                backend.settimeout(30)
                self.request.settimeout(30)
                readers = [self.request, backend]
                while readers and not self.server.stopping.is_set():
                    ready, _, _ = select.select(readers, [], [], 1)
                    for source in ready:
                        destination = backend if source is self.request else self.request
                        chunk = source.recv(65536)
                        if chunk:
                            destination.sendall(chunk)
                        else:
                            destination.shutdown(socket.SHUT_WR)
                            readers.remove(source)
        except OSError:
            pass  # Disconnects are normal when clients cancel streaming requests.


class Bridge(socketserver.ThreadingTCPServer):
    daemon_threads = True

    def __init__(self, backend: tuple[str, int]):
        self.backend = backend
        self.stopping = threading.Event()
        super().__init__(("127.0.0.1", 0), BridgeHandler)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_route(args, process, local_url: str, model: str) -> str:
    deadline = time.monotonic() + args.opentela_timeout
    base_url = None
    last_error = "waiting for local OpenTela identity"
    next_report = 0.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"OpenTela exited with status {process.returncode}; see otela.log.")
        try:
            if base_url is None:
                if urlsplit(args.api_url).path:
                    base_url = args.api_url
                else:
                    identity = get_json(local_url + "/v1/self", use_proxy=False)
                    peer = identity.get("id", "") if isinstance(identity, dict) else ""
                    if not isinstance(peer, str) or not re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{32,128}", peer):
                        raise ValueError("OpenTela has not returned a valid peer ID yet")
                    base_url = f"{args.api_url}/v1/p2p-service/{quote(peer, safe='')}/llm/v1"
                log("OpenTela started; waiting for the gateway route to serve the model")
            if has_model(get_json(base_url + "/models", args.api_key,
                                  use_proxy=not is_local_url(base_url)), model):
                return base_url
            last_error = f"gateway route does not yet list {model}"
        except HTTPError as exc:
            # Do not print the body: remote diagnostics can echo credentials.
            last_error = f"HTTP {exc.code}"
            if exc.code in {401, 402, 403}:
                raise RuntimeError(
                    f"OpenTela returned HTTP {exc.code}; check your API key, account credit, "
                    "and access to this route. An inference API key is not an instance deploy key."
                ) from None
        except (OSError, ValueError, URLError):
            last_error = "route unavailable or invalid JSON response"
        if time.monotonic() >= next_report:
            target = base_url + "/models" if base_url else local_url + "/v1/self"
            log(f"Still waiting for OpenTela at {target}: {last_error}")
            next_report = time.monotonic() + 30
        time.sleep(args.poll_interval)
    raise TimeoutError(f"Timed out verifying OpenTela deployment ({last_error}); "
                       "check otela.log, gateway routing, and bootstrap connectivity.")


def deploy(args, runner: Path, binary: str) -> None:
    state_dir = args.state_dir.expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (state_dir / "deploy.lock").open("a") as lock, contextlib.ExitStack() as stack:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"A deployment already uses {state_dir}.") from None
        log(f"OpenTela state and logs: {state_dir}")
        job_id, host, port = start_backend(args, runner)
        bridge = stack.enter_context(Bridge((host, port)))
        thread = threading.Thread(target=bridge.serve_forever, daemon=True)
        thread.start()
        stack.callback(bridge.shutdown)
        stack.callback(bridge.stopping.set)
        http_port = free_port()
        config = {
            "mode": "node", "role": "worker", "cleanslate": False,
            "port": str(http_port), "tcpport": "0", "udpport": "0",
            "bootstrap": {"sources": args.bootstrap or [BOOTSTRAP]},
            "service": {"name": "llm", "port": str(bridge.server_address[1]),
                        "health_path": "/v1/models"},
        }
        # JSON is a YAML subset; no third-party YAML dependency is needed.
        (state_dir / "cfg.yaml").write_text(json.dumps(config, indent=2) + "\n")
        logfile = stack.enter_context((state_dir / "otela.log").open("a"))
        child_env = {key: value for key, value in os.environ.items() if key != "OPENTELA_API_KEY"}
        process = subprocess.Popen(
            [binary, "start", "--config-dir", str(state_dir)],
            stdin=subprocess.DEVNULL, stdout=logfile, stderr=subprocess.STDOUT,
            env=child_env,
        )
        stack.callback(stop_process, process)
        base_url = wait_for_route(args, process, f"http://127.0.0.1:{http_port}",
                                  slurm.sbatch_served_model_name(runner))
        print(base_url, flush=True)
        log("Deployment verified. Keep this process running; Ctrl-C stops OpenTela and the bridge.")
        while process.poll() is None:
            time.sleep(args.poll_interval)
            job_state(job_id)
        raise RuntimeError(f"OpenTela exited with status {process.returncode}; see {state_dir / 'otela.log'}.")


def stop_process(process) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", default=os.environ.get("OPENTELA_API_KEY"),
                        help="OpenTela inference API key (default: OPENTELA_API_KEY)")
    parser.add_argument("--model", required=True, help="Runner filename without .sbatch")
    parser.add_argument("--cluster", help="Slurm cluster (default: auto-detect)")
    parser.add_argument("--backend-host", help="Override vLLM hostname; {node} expands to the Slurm head node")
    parser.add_argument("--otela", help="OpenTela executable (default: PATH, then download verified v0.2.6)")
    parser.add_argument("--api-url", default="https://api.opentela.ai",
                        help="OpenTela gateway origin or /v1/service/llm URL (optionally ending in /v1)")
    parser.add_argument("--bootstrap", action="append", help="Bootstrap multiaddr or discovery URL; repeatable")
    parser.add_argument("--state-dir", type=Path, help="Persistent peer identity and log directory")
    parser.add_argument("--slurm-start-timeout", type=positive_int, default=1800)
    parser.add_argument("--vllm-ready-timeout", type=positive_int, default=1800)
    parser.add_argument("--opentela-timeout", type=positive_int, default=300)
    parser.add_argument("--poll-interval", type=positive_int, default=5)
    args = parser.parse_args(argv)
    args.api_url = args.api_url.rstrip("/")
    url = urlsplit(args.api_url)
    local_http = url.scheme == "http" and is_local_url(args.api_url)
    if ((url.scheme != "https" and not local_http) or not url.hostname
            or url.username or url.password or url.query or url.fragment
            or url.path not in {"", "/v1/service/llm", "/v1/service/llm/v1"}):
        parser.error("--api-url must be an HTTPS origin or /v1/service/llm URL "
                     "(optionally ending in /v1; HTTP allowed only for localhost)")
    # Preserve an explicit service route: self-hosted gateways may expose it
    # without exposing the peer-specific routing API.
    args.api_url = url._replace(path="/v1/service/llm/v1" if url.path else "").geturl()
    args.api_key = (args.api_key or "").strip()
    if not args.api_key:
        parser.error("provide --api-key or set OPENTELA_API_KEY")
    if any(ord(char) < 33 or ord(char) > 126 for char in args.api_key):
        parser.error("API key must contain only printable non-whitespace ASCII characters")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    def interrupted(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        args.cluster = args.cluster or detect_cluster()
        runner = slurm.supported_runner_path(args.cluster, args.model)
        # Validate the runner before downloading anything or submitting a job.
        slurm.sbatch_job_name(runner)
        slurm.sbatch_vllm_port(runner)
        slurm.sbatch_served_model_name(runner)
        if args.state_dir is None:
            root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
            args.state_dir = root / "pi-slurm-vllm" / "opentela" / args.cluster / args.model
        deploy(args, runner, ensure_otela(args.otela))
    except KeyboardInterrupt:
        log("Stopped OpenTela deployment; Slurm jobs are retained.")
        return 130
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        message = str(exc)
        if args.api_key:
            message = message.replace(args.api_key, "[redacted]")
        log(f"Error: {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
