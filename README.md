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

## Troubleshooting

Check the proxy first:

```bash
curl -sS http://127.0.0.1:8123/status
```

If `backend_base_url` is `null`, the proxy is still waiting for Slurm or vLLM. Check the Slurm job:

```bash
squeue -u "$USER" -n pi-vllm-qwen36-27b
sacct -j <jobid> --format JobID,JobName%30,State,ExitCode,Elapsed,NodeList%30
tail -n 240 logs/pi-vllm-qwen36-27b-<jobid>.err
tail -n 180 logs/pi-vllm-qwen36-27b-<jobid>.out
```

If the log contains:

```text
unsupported GNU version! gcc versions later than 13 are not supported
```

then the worker did not receive `NVCC_APPEND_FLAGS`. Confirm the log has the vLLM Ray env-copy lines shown above. If it does not, the job was launched from an older `slurm/pi-vllm.sbatch` or the environment was changed before vLLM started.

If the log contains:

```text
Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

then the job was launched without the explicit `CUDA_HOME`/`CUDACXX` settings, or Ray did not copy them to workers. The current script sets both before Ray starts and includes them in `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY`.

If the log contains:

```text
gcc: fatal error: cannot execute 'cc1plus'
```

then `nvcc` is seeing a mixed compiler environment. The current script pins `CC`, `CXX`, `CUDAHOSTCXX`, `COMPILER_PATH`, `LIBRARY_PATH`, and `LD_LIBRARY_PATH` to the complete GCCcore 14.3.0 installation and copies those variables to Ray workers.

If the log contains FlashInfer GDN compile errors in files like:

```text
flashinfer/flat/hopper/collective
gdn_prefill_kernel
```

then vLLM is using the FlashInfer GDN prefill backend. The current Slurm script avoids that with `--additional-config '{"gdn_prefill_backend":"triton"}'`.

If the log contains:

```text
Kernel requires a runtime memory allocation, but no allocator was set
```

then the Triton/FLA GDN path was captured in a CUDA graph. The current Slurm script avoids that with `--enforce-eager`.

If Pi returns:

```text
400 Unexpected message role
```

then vLLM is receiving an OpenAI role that the Qwen chat template does not support. The proxy normalizes `/v1/chat/completions` JSON bodies before forwarding: all `system` and `developer` messages are merged into one leading `system` message, `function` is mapped to `tool`, and supported `user`/`assistant`/`tool` messages are preserved. Restart the local proxy or restart Pi after changing `pi_vllm_proxy.py`.

If the job reaches the Slurm time limit before `/v1/models` is available, keep `#SBATCH --time=04:00:00` or increase it. First launch can spend several minutes compiling and warming CUDA graphs.
