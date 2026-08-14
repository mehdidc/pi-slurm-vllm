# Pi vLLM Slurm Proxy

This repository lets Pi use a vLLM server running in a Slurm allocation. Pi
talks to a stable local OpenAI-compatible endpoint at
`http://127.0.0.1:8123/v1`.

The underlying `vllm_proxy.py` is agent-harness-neutral and can serve any
OpenAI-compatible client. Only `extensions/pi-vllm/` contains Pi-specific
integration.

## Installation

Install globally:

```bash
pi install git:github.com/mehdidc/pi-slurm-vllm@v1.0.0
```

Install for one project:

```bash
pi install -l git:github.com/mehdidc/pi-slurm-vllm@v1.0.0
```

Or install from local copy of the repo

```bash
git clone https://github.com/mehdidc/pi-slurm-vllm
pi install /path/pi-slurm-vllm
```

## Usage in Pi 

Start Pi normally:

```bash
pi
```

The extension automatically detects `jureca` or `jupiter` from Slurm, with the
hostname as a fallback. It scans `slurm/<detected-cluster>/*.sbatch` and
registers one `hpc-vllm` model for every runner it finds. No cluster or model
environment variables are needed.

List the discovered catalog with:

```bash
pi --list-models hpc-vllm
```

Select any discovered model inside Pi, for instance:

```text
/model hpc-vllm/Kimi-K3
```

The extension starts the matching proxy lazily on the first request. Switching
to another discovered model also switches the extension-owned proxy to that
model; existing Slurm jobs remain available for later reuse.

Useful commands:

```text
/vllm-start
/vllm-status
/vllm-stop
```

`/vllm-stop` stops only the proxy created by the current Pi process. It does not
cancel the Slurm job.

The proxy port can host only one cluster/model selection at a time. The
extension can switch a proxy process it owns, but it will not stop an unrelated
proxy already using that port.

## Stress testing

`stress_test.py` sends 1,288 examples from the official
[OpenAI GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k) in
`benchmarks/math_problems.jsonl` concurrently to the same OpenAI-compatible
proxy endpoint used by Pi. The bundled deterministic subset contains test
rows 0 through 1,287 in their published order, including each reference
answer. Start the desired model from Pi first, then run:

```bash
python3 stress_test.py --parallel 16 --model Kimi-K3 --reasoning-effort low
```

The summary reports successful requests per second, aggregate prefill
(input-token) and decode (output-token) throughput, and mean/p50/p95/p99 request
latency. It also reports p50/p95/p99 of per-request output-token rates, calculated
as each response's `usage.completion_tokens` divided by that request's total
latency. Because responses are non-streaming, this request latency includes
queueing and prefill as well as decoding. Prefill and decode may overlap inside
vLLM, so the aggregate figures are end-to-end workload rates over the same
measured wall-clock interval rather than isolated engine-phase timings. For
managed `--sbatch` runs, the summary additionally
locates the Slurm stdout file and reports the mean and peak of vLLM's native
periodic prompt/generation throughput samples emitted during the measured
batch, including the number of running requests observed at each peak. Save
every response and its timing information with:

```bash
python3 stress_test.py \
  --parallel 16 \
  --model Kimi-K3 \
  --results benchmarks/results.jsonl
```

Use `--requests N` for a shorter run, `--max-tokens N` to control response
length, and `--warmup 0` to disable the default unmeasured warm-up request.
Normally `--requests N` takes at most the first N rows. Add `--replacement` to
sample exactly N requests with replacement, even when the problem file has
fewer rows:

```bash
python3 stress_test.py \
  --problems benchmarks/one_problem.jsonl \
  --requests 100 \
  --replacement \
  --seed 42
```

`--seed` is optional and makes replacement sampling reproducible.

The stress test can also own the complete Slurm lifecycle. Pass a runner with
`--sbatch`; it submits a fresh job, waits for the allocation and vLLM readiness,
runs the same benchmark and throughput report directly against vLLM, then
cancels the whole job and waits for it to disappear from Slurm before exiting:

```bash
python3 stress_test.py \
  --sbatch slurm/jupiter/Kimi-K3.sbatch \
  --parallel 16
```

With `--sbatch`, the request model defaults to the runner's literal
`SERVED_MODEL_NAME`; an explicit `--model` still overrides it. Without
`--sbatch`, the existing `Kimi-K3` default is retained.

Startup, readiness, shutdown, and polling limits can be adjusted with
`--slurm-start-timeout`, `--vllm-ready-timeout`, `--slurm-stop-timeout`, and
`--poll-interval`. The submitted job is also cancelled if startup, warm-up, or
stress testing fails or the process is interrupted. The summary includes the
vLLM readiness time measured from when the job first reaches `RUNNING`; time
spent waiting in the Slurm queue is excluded.


## Runner contract

Supported model runners live at:

```text
slurm/<cluster>/<model>.sbatch
```

For example, for cluster `jupiter` and model `Kimi-K3` uses
`slurm/jupiter/Kimi-K3.sbatch.sbatch`.

Each runner owns the complete cluster and vLLM configuration: allocation,
environment, model path, served model name, parallelism, parsers, context
length, and port. The proxy only selects the runner, submits or reuses its job,
waits for vLLM, and forwards requests.

A supported runner must:

- be named `slurm/<cluster>/<model>.sbatch`;
- define a cluster-unique `#SBATCH --job-name=...`, which the proxy uses to
  find a reusable job;
- define a literal `VLLM_PORT=<port>`, which the proxy uses to reach vLLM;
- define a literal `SERVED_MODEL_NAME=<name>`, which managed stress tests use
  as their default OpenAI request model;
- pass a literal `--max-model-len VALUE`, which the Pi extension uses for the
  model's context window;
- serve the model under the same name as the runner filename.

To support another cluster/model pair, add its dedicated `.sbatch` file. No
proxy code change is needed.


## Manual proxy mode

Run the generic proxy with a supported cluster/model pair:

```bash
python3 vllm_proxy.py --model Kimi-K3
```

The first request submits or reuses the job declared by the selected runner.
Unsupported pairs fail immediately and print the available runners.

## Running Pi from a laptop

If the proxy runs on a cluster login node and Pi runs on your laptop, forward
the proxy port:

```bash
ssh -L 8123:127.0.0.1:8123 <cluster-login>
```

Keep Pi's base URL as `http://127.0.0.1:8123/v1`.

By default, the proxy assumes the login node can directly reach the compute
node. If the cluster requires an SSH tunnel:

```bash
python3 vllm_proxy.py \
  --model Kimi-K3 \
  --ssh-tunnel
```

Use `--ssh-host <cluster-login>` as well if the tunnel needs a specific SSH
target.
