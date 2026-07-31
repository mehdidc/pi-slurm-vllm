#!/usr/bin/env python3
"""Run concurrent math requests against the OpenAI-compatible Pi proxy."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_URL = "http://127.0.0.1:8123/v1/chat/completions"
DEFAULT_PROBLEMS = Path(__file__).resolve().parent / "benchmarks" / "math_problems.jsonl"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


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


def print_summary(results: list[dict[str, Any]], elapsed: float, parallel: int) -> None:
    successes = [result for result in results if result["ok"]]
    failures = [result for result in results if not result["ok"]]
    latencies = [float(result["latency_seconds"]) for result in successes]
    prompt_tokens = sum(int(result["prompt_tokens"]) for result in successes)
    completion_tokens = sum(int(result["completion_tokens"]) for result in successes)

    print("\nStress-test summary")
    print(f"  concurrency:             {parallel}")
    print(f"  requests:                {len(results)}")
    print(f"  successful / failed:     {len(successes)} / {len(failures)}")
    print(f"  wall time:               {elapsed:.2f} s")
    print(f"  successful requests/s:   {len(successes) / elapsed:.3f}")
    print(f"  input tokens:            {prompt_tokens}")
    print(f"  output tokens:           {completion_tokens}")
    print(f"  output tokens/s:         {completion_tokens / elapsed:.2f}")
    print(f"  total tokens/s:          {(prompt_tokens + completion_tokens) / elapsed:.2f}")
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
    parser.add_argument("--model", default="Kimi-K3", help="Served model name (default: Kimi-K3)")
    parser.add_argument("--problems", type=Path, default=DEFAULT_PROBLEMS, help="Input JSONL problem set")
    parser.add_argument("--results", type=Path, help="Optional JSONL file for per-request results")
    parser.add_argument("--max-tokens", type=positive_int, default=1024, help="Maximum output tokens per request")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", choices=("low", "high", "max"))
    parser.add_argument("--timeout", type=positive_int, default=900, help="Per-request timeout in seconds")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--warmup", type=int, choices=(0, 1), default=1, help="Run one unmeasured request first")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    problems = load_problems(args.problems)
    selected = problems[: args.requests] if args.requests else problems

    if args.warmup:
        print("Running one warm-up request...", file=sys.stderr)
        warmup = send_request(args, selected[0], 0)
        if not warmup["ok"]:
            print(f"Warm-up failed: {warmup['error']}", file=sys.stderr)
            return 2

    print(
        f"Sending {len(selected)} requests to {args.url} with concurrency {args.parallel}...",
        file=sys.stderr,
    )
    results, elapsed = run_batch(args, selected, show_progress=True)
    print_summary(results, elapsed, args.parallel)

    if args.results:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        with args.results.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"Wrote detailed results to {args.results}")

    return 1 if any(not result["ok"] for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
