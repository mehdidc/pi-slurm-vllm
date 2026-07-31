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

Or try the repository directly:

```bash
pi -e /path/pi-slurm-vllm
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
- pass a literal `--max-model-len VALUE`, which the Pi extension uses for the
  model's context window;
- serve the model under the same name as the runner filename.

To support another cluster/model pair, add its dedicated `.sbatch` file. No
proxy code change is needed.


## Manual proxy mode

Run the generic proxy with a supported cluster/model pair:

```bash
python3 vllm_proxy.py \
  --cluster jupiter \
  --model Kimi-K3.sbatch
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
  --cluster jureca \
  --model Kimi-K3.sbatch \
  --ssh-tunnel
```

Use `--ssh-host <cluster-login>` as well if the tunnel needs a specific SSH
target.
