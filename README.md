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

## What Was Fixed

The recurring Qwen3.6 27B failures were not Pi-side model registration problems. Pi could not connect because the Slurm vLLM job was failing before the OpenAI server reached `/v1/models`.

The main failure chain was:

1. The proxy submitted the Slurm job correctly, but vLLM was still starting when Pi tried to use it.
2. Ray workers did not reliably inherit the environment needed by vLLM and FlashInfer.
3. FlashInfer 0.6.6 JIT compilation used `nvcc` on the worker side. The cluster exposes GCC 14.3, while CUDA 12 rejects host compilers newer than GCC 13 unless `nvcc` receives `-allow-unsupported-compiler`.
4. The old `FLASHINFER_EXTRA_CUDAFLAGS=-allow-unsupported-compiler` setting did not help here because this installed FlashInfer version does not read that variable.
5. Some cache locations under shared filesystems could also stall or preserve failed FlashInfer builds, so vLLM startup could hang or repeatedly reuse a bad cache state.
6. Qwen3-Next's default GDN prefill backend tried the FlashInfer GDN JIT path on GH200 and failed in generated CUTLASS/CUTE code. Switching to Triton/FLA avoided that compile path.
7. The Triton/FLA GDN path then failed under CUDA graph capture with `Kernel requires a runtime memory allocation, but no allocator was set`, so this model is served with eager execution.

The fix is in `slurm/pi-vllm.sbatch`:

```bash
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-} -allow-unsupported-compiler"
export GCC_HOME=${GCC_HOME:-/e/software/default/stages/2026/software/GCCcore/14.3.0}
export CC=${CC:-${GCC_HOME}/bin/gcc}
export CXX=${CXX:-${GCC_HOME}/bin/g++}
export CUDAHOSTCXX=${CUDAHOSTCXX:-${CXX}}
export CUDA_HOME=${CUDA_HOME:-/e/software/default/stages/2025/software/CUDA/12}
export CUDACXX=${CUDACXX:-${CUDA_HOME}/bin/nvcc}
export VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY="FLASHINFER_,NVCC_,XDG_,TRITON_,TORCHINDUCTOR_"
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="CUDA_HOME,CUDACXX,GCC_HOME,CC,CXX,CUDAHOSTCXX,COMPILER_PATH,LIBRARY_PATH,LD_LIBRARY_PATH,PATH"
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-3600}
```

`NVCC_APPEND_FLAGS` is consumed by `nvcc` itself, so it reaches FlashInfer JIT even though FlashInfer does not read `FLASHINFER_EXTRA_CUDAFLAGS`. The `VLLM_RAY_EXTRA_ENV_*` settings make vLLM copy the required variables from the driver into Ray workers. The current logs should include lines like:

```text
Env var prefixes to copy: [..., 'FLASHINFER_', 'NVCC_', ...]
Copying the following environment variables to workers: [..., 'NVCC_APPEND_FLAGS', 'FLASHINFER_WORKSPACE_BASE', ...]
```

The script also puts runtime/JIT caches on node-local storage before Ray and vLLM start:

```bash
CACHE_ROOT=${PI_VLLM_CACHE_ROOT:-${SLURM_TMPDIR:-/tmp}/pi-vllm-${USER:-user}}
export FLASHINFER_WORKSPACE_BASE=${CACHE_ROOT}/flashinfer_workspace
export VLLM_CACHE_ROOT=${CACHE_ROOT}/vllm
export VLLM_CONFIG_ROOT=${CACHE_ROOT}/vllm_config
export TRITON_CACHE_DIR=${CACHE_ROOT}/triton_cache_dir
export TORCHINDUCTOR_CACHE_DIR=${CACHE_ROOT}/torchinductor_cache
export XDG_CACHE_HOME=${CACHE_ROOT}/xdg_cache
```

This avoids shared filesystem stalls and avoids reusing failed FlashInfer build artifacts from `~/.cache`.

The serve command also passes two Qwen3-Next-specific options:

```bash
--enforce-eager \
--additional-config '{"gdn_prefill_backend":"triton"}'
```

`gdn_prefill_backend=triton` prevents vLLM from selecting the FlashInfer GDN prefill kernel. `--enforce-eager` disables CUDA graph capture for this model, which avoids the Triton runtime allocator failure seen during GDN prefill.

The verified good path is:

```bash
curl http://127.0.0.1:8123/status
curl http://127.0.0.1:8123/v1/models
```

The status should show a running Slurm job, a compute-node backend URL, and no `last_error`.

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
