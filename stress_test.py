#!/usr/bin/env python3
"""Run concurrent math requests against the OpenAI-compatible Pi proxy."""

from __future__ import annotations

import argparse
import json
import math
import re
import signal
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from vllm_proxy import (
    first_hostname,
    get_job_state,
    resolve_host,
    sbatch_served_model_name,
    sbatch_vllm_port,
    submit_job,
    wait_for_backend,
    wait_for_running,
)


DEFAULT_URL = "http://127.0.0.1:8123/v1/chat/completions"
DEFAULT_MODEL = "Kimi-K3"
DEFAULT_PROBLEMS = Path(__file__).resolve().parent / "benchmarks" / "math_problems.jsonl"
VLLM_THROUGHPUT_RE = re.compile(
    r"Avg prompt throughput:\s*([0-9]+(?:\.[0-9]+)?) tokens/s,\s*"
    r"Avg generation throughput:\s*([0-9]+(?:\.[0-9]+)?) tokens/s,\s*"
    r"Running:\s*(\d+) reqs"
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def submit_slurm_backend(args: argparse.Namespace) -> tuple[str, int, str]:
    """Submit a runner and return its job ID, port, and served model name."""
    sbatch_path = args.sbatch.expanduser().resolve()
    if not sbatch_path.is_file():
        raise FileNotFoundError(f"Slurm runner does not exist: {sbatch_path}")

    vllm_port = sbatch_vllm_port(sbatch_path)
    served_model_name = sbatch_served_model_name(sbatch_path)
    print(f"Submitting Slurm runner {sbatch_path}...", file=sys.stderr)
    job_id = submit_job(str(sbatch_path))
    print(f"Submitted Slurm job {job_id}; waiting for its allocation...", file=sys.stderr)
    return job_id, vllm_port, served_model_name


def wait_for_slurm_backend(
    args: argparse.Namespace,
    job_id: str,
    vllm_port: int,
) -> tuple[str, float]:
    """Return the backend URL and readiness time measured after RUNNING."""
    _, nodes = wait_for_running(job_id, args.slurm_start_timeout, args.poll_interval)
    running_started = time.perf_counter()
    head_node = first_hostname(nodes)
    host = resolve_host(head_node) if args.resolve_node_ip else head_node
    backend_url = f"http://{host}:{vllm_port}"
    print(f"Job {job_id} is running; waiting for vLLM at {backend_url}...", file=sys.stderr)
    wait_for_backend(backend_url, args.vllm_ready_timeout, args.poll_interval)
    ready_seconds = time.perf_counter() - running_started
    print(
        f"vLLM is ready for stress testing (job {job_id}) after "
        f"{ready_seconds:.2f}s in RUNNING state.",
        file=sys.stderr,
    )
    return backend_url, ready_seconds


def stop_slurm_job(job_id: str, timeout_s: int, poll_s: int) -> None:
    """Cancel the entire allocation and wait until Slurm removes it from squeue."""
    print(f"Cancelling Slurm job {job_id}...", file=sys.stderr)
    proc = subprocess.run(
        ["scancel", "--full", job_id],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0 and get_job_state(job_id) is not None:
        raise RuntimeError(
            f"scancel {job_id} failed with {proc.returncode}: {proc.stderr.strip()}"
        )

    deadline = time.monotonic() + timeout_s
    last_state: str | None = None
    while time.monotonic() < deadline:
        job = get_job_state(job_id)
        if job is None:
            print(f"Slurm job {job_id} has stopped.", file=sys.stderr)
            return
        last_state = job[0]
        time.sleep(poll_s)
    raise TimeoutError(
        f"Timed out waiting for Slurm job {job_id} to stop; last state: {last_state}"
    )


def slurm_stdout_path(job_id: str) -> Path | None:
    """Return Slurm's resolved stdout path for a live job, if available."""
    proc = subprocess.run(
        ["scontrol", "show", "job", "-o", job_id],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        return None
    match = re.search(r"(?:^|\s)StdOut=(\S+)", proc.stdout)
    if not match or match.group(1) in {"/dev/null", "(null)"}:
        return None
    return Path(match.group(1))


def vllm_log_throughput_samples(
    path: Path,
    offset: int = 0,
) -> list[tuple[float, float, int]]:
    """Parse vLLM prompt/generation throughput samples appended after offset."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(offset if 0 <= offset <= size else 0)
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    return [
        (float(match.group(1)), float(match.group(2)), int(match.group(3)))
        for match in VLLM_THROUGHPUT_RE.finditer(text)
    ]


def load_problems(path: Path) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                problem = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
            if not isinstance(problem, dict) or not isinstance(problem.get("prompt"), str):
                raise ValueError(f"{path}:{line_number} must contain an object with a prompt string")
            problems.append(problem)
    if not problems:
        raise ValueError(f"No problems found in {path}")
    return problems


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def make_payload(args: argparse.Namespace, problem: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": problem["prompt"] + " Show a concise derivation and state the final answer clearly.",
            }
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    if args.reasoning_effort:
        payload["reasoning_effort"] = args.reasoning_effort
    return payload


def send_request(
    args: argparse.Namespace,
    problem: dict[str, Any],
    sequence: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    request = Request(
        args.url,
        data=json.dumps(make_payload(args, problem)).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {args.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=args.timeout) as response:
            response_body = response.read()
        decoded = json.loads(response_body)
        usage = decoded.get("usage") or {}
        choices = decoded.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        return {
            "sequence": sequence,
            "problem_id": problem.get("id", sequence),
            "ok": True,
            "latency_seconds": time.perf_counter() - started,
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
            "response": message.get("content", ""),
        }
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        error = f"HTTP {exc.code}: {detail}"
    except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "sequence": sequence,
        "problem_id": problem.get("id", sequence),
        "ok": False,
        "latency_seconds": time.perf_counter() - started,
        "error": error,
    }


def run_batch(
    args: argparse.Namespace,
    problems: list[dict[str, Any]],
    show_progress: bool,
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(send_request, args, problem, sequence): sequence
            for sequence, problem in enumerate(problems, 1)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            if show_progress:
                status = "ok" if result["ok"] else "FAILED"
                print(
                    f"[{completed:>3}/{len(problems)}] {result['problem_id']}: "
                    f"{status} in {result['latency_seconds']:.2f}s",
                    file=sys.stderr,
                )
    return sorted(results, key=lambda item: item["sequence"]), time.perf_counter() - started


def print_summary(
    results: list[dict[str, Any]],
    elapsed: float,
    parallel: int,
    vllm_ready_seconds: float | None = None,
    vllm_engine_samples: list[tuple[float, float, int]] | None = None,
) -> None:
    successes = [result for result in results if result["ok"]]
    failures = [result for result in results if not result["ok"]]
    latencies = [float(result["latency_seconds"]) for result in successes]
    per_request_output_rates = [
        int(result["completion_tokens"]) / float(result["latency_seconds"])
        for result in successes
        if float(result["latency_seconds"]) > 0
    ]
    prompt_tokens = sum(int(result["prompt_tokens"]) for result in successes)
    completion_tokens = sum(int(result["completion_tokens"]) for result in successes)

    print("\nStress-test summary")
    if vllm_ready_seconds is not None:
        print(f"  vLLM ready time:          {vllm_ready_seconds:.2f} s (from job RUNNING)")
    print(f"  concurrency:             {parallel}")
    print(f"  requests:                {len(results)}")
    print(f"  successful / failed:     {len(successes)} / {len(failures)}")
    print(f"  wall time:               {elapsed:.2f} s")
    print(f"  successful requests/s:   {len(successes) / elapsed:.3f}")
    print(f"  input tokens:            {prompt_tokens}")
    print(f"  output tokens:           {completion_tokens}")
    print(f"  prefill throughput:      {prompt_tokens / elapsed:.2f} input tokens/s")
    print(f"  decode throughput:       {completion_tokens / elapsed:.2f} output tokens/s")
    print(f"  total tokens/s:          {(prompt_tokens + completion_tokens) / elapsed:.2f}")
    if per_request_output_rates:
        print(
            "  request output tok/s "
            f"p50/p95/p99: {percentile(per_request_output_rates, 0.50):.2f} / "
            f"{percentile(per_request_output_rates, 0.95):.2f} / "
            f"{percentile(per_request_output_rates, 0.99):.2f}"
        )
    if vllm_engine_samples:
        engine_prefill = [sample[0] for sample in vllm_engine_samples]
        engine_decode = [sample[1] for sample in vllm_engine_samples]
        prefill_peak = max(vllm_engine_samples, key=lambda sample: sample[0])
        decode_peak = max(vllm_engine_samples, key=lambda sample: sample[1])
        print(f"  vLLM engine samples:     {len(vllm_engine_samples)}")
        print(
            f"  engine prefill avg/peak: {statistics.fmean(engine_prefill):.2f} / "
            f"{prefill_peak[0]:.2f} tokens/s ({prefill_peak[2]} running reqs at peak)"
        )
        print(
            f"  engine decode avg/peak:  {statistics.fmean(engine_decode):.2f} / "
            f"{decode_peak[1]:.2f} tokens/s ({decode_peak[2]} running reqs at peak)"
        )
    elif vllm_engine_samples is not None:
        print("  vLLM engine samples:     none emitted during measured batch")
    if latencies:
        print(f"  latency mean:            {statistics.fmean(latencies):.2f} s")
        print(f"  latency p50 / p95 / p99: {percentile(latencies, 0.50):.2f} / "
              f"{percentile(latencies, 0.95):.2f} / {percentile(latencies, 0.99):.2f} s")
    if failures:
        print("\nFailures:", file=sys.stderr)
        for result in failures:
            print(f"  {result['problem_id']}: {result['error']}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel", type=positive_int, default=8, help="Concurrent requests (default: 8)")
    parser.add_argument("--requests", type=positive_int, help="Number of problems to run (default: all 1,288)")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Chat completions URL (default: {DEFAULT_URL})")
    parser.add_argument(
        "--model",
        help=(
            "Served model name (default: SERVED_MODEL_NAME from --sbatch, "
            f"otherwise {DEFAULT_MODEL})"
        ),
    )
    parser.add_argument("--problems", type=Path, default=DEFAULT_PROBLEMS, help="Input JSONL problem set")
    parser.add_argument("--results", type=Path, help="Optional JSONL file for per-request results")
    parser.add_argument("--max-tokens", type=positive_int, default=1024, help="Maximum output tokens per request")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", choices=("low", "high", "max"))
    parser.add_argument("--timeout", type=positive_int, default=900, help="Per-request timeout in seconds")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--warmup", type=int, choices=(0, 1), default=1, help="Run one unmeasured request first")
    parser.add_argument(
        "--sbatch",
        type=Path,
        help=(
            "Submit this vLLM runner before testing, use its allocated node directly, "
            "and cancel the job before exiting"
        ),
    )
    parser.add_argument(
        "--slurm-start-timeout",
        type=positive_int,
        default=1800,
        help="Seconds to wait for a submitted job to start (default: 1800)",
    )
    parser.add_argument(
        "--vllm-ready-timeout",
        type=positive_int,
        default=1800,
        help="Seconds to wait for vLLM readiness (default: 1800)",
    )
    parser.add_argument(
        "--slurm-stop-timeout",
        type=positive_int,
        default=300,
        help="Seconds to wait for the job to disappear after cancellation (default: 300)",
    )
    parser.add_argument(
        "--poll-interval",
        type=positive_int,
        default=10,
        help="Slurm and vLLM readiness polling interval in seconds (default: 10)",
    )
    parser.add_argument("--resolve-node-ip", action="store_true", default=True)
    parser.add_argument("--no-resolve-node-ip", dest="resolve_node_ip", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.sbatch and args.model is None:
        args.model = DEFAULT_MODEL
    job_id: str | None = None
    vllm_ready_seconds: float | None = None
    exit_code = 0
    interrupted_signal: int | None = None
    previous_sigterm_handler: Any = None
    previous_sigint_handler: Any = None
    slurm_log_path: Path | None = None
    slurm_log_offset = 0

    def handle_sigterm(signum: int, _frame: Any) -> None:
        nonlocal interrupted_signal
        interrupted_signal = signum
        raise KeyboardInterrupt

    if args.sbatch:
        previous_sigterm_handler = signal.signal(signal.SIGTERM, handle_sigterm)
        previous_sigint_handler = signal.getsignal(signal.SIGINT)

    try:
        problems = load_problems(args.problems)
        selected = problems[: args.requests] if args.requests else problems

        if args.sbatch:
            job_id, vllm_port, runner_model = submit_slurm_backend(args)
            if args.model is None:
                args.model = runner_model
                print(f"Using model {args.model} from the Slurm runner.", file=sys.stderr)
            backend_url, vllm_ready_seconds = wait_for_slurm_backend(
                args, job_id, vllm_port
            )
            args.url = f"{backend_url}/v1/chat/completions"

        if args.warmup:
            print("Running one warm-up request...", file=sys.stderr)
            warmup = send_request(args, selected[0], 0)
            if not warmup["ok"]:
                print(f"Warm-up failed: {warmup['error']}", file=sys.stderr)
                exit_code = 2
            else:
                print("Warm-up completed.", file=sys.stderr)

        if exit_code == 0:
            if job_id is not None:
                slurm_log_path = slurm_stdout_path(job_id)
                if slurm_log_path is not None:
                    try:
                        slurm_log_offset = slurm_log_path.stat().st_size
                    except OSError:
                        slurm_log_path = None
            print(
                f"Sending {len(selected)} requests to {args.url} with concurrency {args.parallel}...",
                file=sys.stderr,
            )
            results, elapsed = run_batch(args, selected, show_progress=True)
            engine_samples = (
                vllm_log_throughput_samples(slurm_log_path, slurm_log_offset)
                if slurm_log_path is not None
                else None
            )
            print_summary(
                results,
                elapsed,
                args.parallel,
                vllm_ready_seconds,
                engine_samples,
            )

            if args.results:
                args.results.parent.mkdir(parents=True, exist_ok=True)
                with args.results.open("w", encoding="utf-8") as handle:
                    for result in results:
                        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(f"Wrote detailed results to {args.results}")

            exit_code = 1 if any(not result["ok"] for result in results) else 0
    except KeyboardInterrupt:
        print("Stress test interrupted.", file=sys.stderr)
        exit_code = 128 + (interrupted_signal or signal.SIGINT)
    except Exception as exc:
        print(f"Stress test failed: {exc}", file=sys.stderr)
        exit_code = 2
    finally:
        if job_id is not None:
            # Once cancellation starts, do not let a second terminal signal leave
            # the allocation behind midway through cleanup.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            try:
                stop_slurm_job(job_id, args.slurm_stop_timeout, args.poll_interval)
            except Exception as exc:
                print(f"Failed to stop Slurm job {job_id}: {exc}", file=sys.stderr)
                exit_code = 2
        if args.sbatch:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)
            signal.signal(signal.SIGINT, previous_sigint_handler)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
