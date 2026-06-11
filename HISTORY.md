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
