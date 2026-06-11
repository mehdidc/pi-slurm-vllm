# Pi vLLM Slurm Proxy

This repo makes Pi use a vLLM server running inside a Slurm allocation.

Pi talks to a stable local OpenAI-compatible endpoint:

```text
http://127.0.0.1:8123/v1
```

The proxy behind that endpoint checks for an existing `pi-vllm-qwen36-27b` Slurm job. If none exists, it submits `slurm/pi-vllm.sbatch`, waits for the job to become `RUNNING`, waits for vLLM to answer `/v1/models`, and forwards Pi requests to vLLM.

## Files

- `extensions/pi-vllm/index.ts`: Pi extension entry point (registers provider, starts proxy, exposes `/vllm-*` commands).
- `pi_vllm_proxy.py`: local OpenAI-compatible proxy and Slurm lifecycle wrapper.
- `slurm/pi-vllm.sbatch`: Slurm job script for launching vLLM.
- `package.json`: Pi package manifest.

## Installation

### As a Pi Package

Install globally (available in all projects):

```bash
pi install git:github.com/mehdidc/pi-slurm-vllm@v1.0.0
```

Install for a specific project (written to `.pi/settings.json`):

```bash
pi install -l git:github.com/mehdidc/pi-slurm-vllm@v1.0.0
```

Try without installing:

```bash
pi -e /path/pi-slurm-vllm
```

### From This Repository

No separate install is needed. Starting Pi from this directory auto-discovers the extension via the `extensions/` convention directory declared in `package.json`.


## Use Directly Inside Pi

When you start Pi from this project (or after installing the package), the extension registers the `hpc-vllm/Qwen3.6-27B-FP8` model and starts the local proxy inside the Pi process. The Slurm job is lazy: it is submitted only when Pi first sends a request to the model.

Start Pi from this repo, then select:

```text
/model hpc-vllm/Qwen3.6-27B-FP8
```

Useful commands inside Pi:

```text
/vllm-start
/vllm-status
/vllm-stop
```

`/vllm-stop` stops only the local proxy process. It does not cancel the Slurm job.

## Manual Proxy Mode

From this repo:

```bash
mkdir -p logs
python3 pi_vllm_proxy.py
```

The first request from Pi will submit or reuse the Slurm job.

To submit a different model path while keeping the same script:

```bash
PI_VLLM_MODEL=/e/data1/datasets/products/mmlaion/shared/models/Qwen/Qwen3.6-27B-FP8/ \
PI_VLLM_SERVED_MODEL_NAME=Qwen3.6-27B-FP8 \
python3 pi_vllm_proxy.py --model-id Qwen3.6-27B-FP8
```

## Optional Manual Pi Config

The project-local extension registers the model automatically. Use this only if you want to run the manual proxy without the extension. Merge `models.example.json` into:

```text
~/.pi/agent/models.json
```

Then select the model in Pi with:

```text
/model hpc-vllm/Qwen3.6-27B-FP8
```

## Running Pi from a Laptop

If Pi runs on your laptop and the proxy runs on the cluster login node, forward the proxy port:

```bash
ssh -L 8123:127.0.0.1:8123 <cluster-login>
```

Then keep `baseUrl` as:

```text
http://127.0.0.1:8123/v1
```

## Direct vs SSH Tunnel to Compute Node

By default the proxy assumes the login node can directly reach the compute node vLLM port. If your cluster requires SSH tunneling to the compute node, start the proxy with:

```bash
python3 pi_vllm_proxy.py --ssh-tunnel
```

If the tunnel must go through a specific host:

```bash
python3 pi_vllm_proxy.py --ssh-tunnel --ssh-host <cluster-login>
```

