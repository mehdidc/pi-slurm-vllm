#!/bin/bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --job-name=pi-vllm-qwen36-27b
#SBATCH --output=/e/project1/laionize/cherti1/pi-vllm/logs/%x-%j.out
#SBATCH --error=/e/project1/laionize/cherti1/pi-vllm/logs/%x-%j.err

set -euo pipefail
set -x
UV_PROJECT=/e/project1/laionize/cherti1/learning_to_optimize
export UV_PROJECT
NET_IFACE=ib0
export NCCL_SOCKET_IFNAME=$NET_IFACE
export GLOO_SOCKET_IFNAME=$NET_IFACE
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=120
export PYTORCH_ALLOC_CONF=expandable_segments:True
export VLLM_LOGGING_LEVEL=INFO
# Pin a complete host compiler toolchain for nvcc. Without this, gcc may come
# from the Python/conda env while cc1plus lives in GCCcore, causing FlashInfer
# JIT to fail with "cannot execute 'cc1plus'".
export GCC_HOME=${GCC_HOME:-/e/software/default/stages/2026/software/GCCcore/14.3.0}
export CC=${CC:-${GCC_HOME}/bin/gcc}
export CXX=${CXX:-${GCC_HOME}/bin/g++}
export CUDAHOSTCXX=${CUDAHOSTCXX:-${CXX}}
export COMPILER_PATH="${GCC_HOME}/libexec/gcc/aarch64-unknown-linux-gnu/14.3.0:${COMPILER_PATH:-}"
export LIBRARY_PATH="${GCC_HOME}/lib64:${GCC_HOME}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${GCC_HOME}/lib64:${GCC_HOME}/lib:${LD_LIBRARY_PATH:-}"
export PATH="${GCC_HOME}/bin:${PATH}"
# FlashInfer JIT needs a CUDA toolkit path on Ray workers for first real inference.
export CUDA_HOME=${CUDA_HOME:-/e/software/default/stages/2025/software/CUDA/12}
export CUDACXX=${CUDACXX:-${CUDA_HOME}/bin/nvcc}
export PATH="${CUDA_HOME}/bin:${PATH}"
# FlashInfer JIT invokes nvcc; this cluster exposes GCC 14.3 while CUDA 12 rejects GCC > 13 by default.
# nvcc reads NVCC_APPEND_FLAGS directly; FlashInfer 0.6.6 does not read FLASHINFER_EXTRA_CUDAFLAGS.
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-} -allow-unsupported-compiler"
export VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY="FLASHINFER_,NVCC_,XDG_,TRITON_,TORCHINDUCTOR_"
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="CUDA_HOME,CUDACXX,GCC_HOME,CC,CXX,CUDAHOSTCXX,COMPILER_PATH,LIBRARY_PATH,LD_LIBRARY_PATH,PATH"
# First launch can spend many minutes compiling FlashInfer kernels and loading weights.
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-3600}
unset CUDA_LAUNCH_BLOCKING
unset VLLM_TRACE_FUNCTION

export HF_HUB_OFFLINE=1
export HF_HOME=/e/data1/datasets/playground/mmlaion/shared/cherti1/hf_cache

# Use node-local storage for runtime/JIT caches. /e/scratch can stall on some nodes,
# and home/project cache paths are shared across jobs and can retain failed FlashInfer builds.
CACHE_ROOT=${PI_VLLM_CACHE_ROOT:-${SLURM_TMPDIR:-/tmp}/pi-vllm-${USER:-user}}
export FLASHINFER_WORKSPACE_BASE=${CACHE_ROOT}/flashinfer_workspace
export VLLM_CACHE_ROOT=${CACHE_ROOT}/vllm
export VLLM_CONFIG_ROOT=${CACHE_ROOT}/vllm_config
export TRITON_CACHE_DIR=${CACHE_ROOT}/triton_cache_dir
export TRITON_DUMP_DIR=${CACHE_ROOT}/triton_dump_dir
export TRITON_OVERRIDE_DIR=${CACHE_ROOT}/triton_override_dir
export TRITON_KERNEL_DUMP=${CACHE_ROOT}/triton_kernel_dump
export TORCHINDUCTOR_CACHE_DIR=${CACHE_ROOT}/torchinductor_cache
export XDG_CACHE_HOME=${CACHE_ROOT}/xdg_cache
mkdir -p "$FLASHINFER_WORKSPACE_BASE" "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT" "$TRITON_CACHE_DIR" "$TRITON_DUMP_DIR" "$TRITON_OVERRIDE_DIR" "$TRITON_KERNEL_DUMP" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"

# Keep W&B local/offline. Set WANDB_API_KEY in the submission environment if needed.
export WANDB_MODE=offline
export NVTE_DEBUG_LEVEL=2

export VLLM_HOST_IP=$(ip -o -4 addr show "$NET_IFACE" | awk '{print $4}' | cut -d/ -f1)
if [ -z "$VLLM_HOST_IP" ]; then
  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
fi

NNODES=${SLURM_JOB_NUM_NODES}
export NNODES

echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
unset VLLM_ENABLE_V1_MULTIPROCESSING
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
export VLLM_USE_RAY_COMPILED_DAG=0
unset VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE
unset VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM
unset VLLM_USE_RAY_WRAPPED_PP_COMM
export RAY_memory_monitor_refresh_ms=0
export RAY_LOGGING_LEVEL=DEBUG
export HYDRA_FULL_ERROR=1
export RAY_CGRAPH_get_timeout=1800

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
port=6379
export MASTER_ADDR="$head_node"
MASTER_IP=$(getent hosts "$MASTER_ADDR" | awk '{ print $1 }')
export MASTER_ADDR=$MASTER_IP
export MASTER_PORT=$port

echo "Starting Ray HEAD at $MASTER_ADDR:$MASTER_PORT"
until nvidia-smi > /dev/null 2>&1; do
  echo "Waiting for GPU visibility..."
  sleep 2
done

srun --nodes=1 --ntasks=1 -w "$head_node" bash -lc '
    IFACE=ib0
    export VLLM_HOST_IP=$(ip -o -4 addr show "$IFACE" | awk "{print \$4}" | cut -d/ -f1)
    if [ -z "$VLLM_HOST_IP" ]; then
      export VLLM_HOST_IP=$(hostname -I | awk "{print \$1}")
    fi
    uv --project "$UV_PROJECT" run ray start --head --node-ip-address="$VLLM_HOST_IP" \
      --port='"$port"' --num-cpus '"$SLURM_CPUS_PER_TASK"' --num-gpus '"$SLURM_GPUS_PER_NODE"' --block
  ' &

sleep 10

worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "Starting WORKER $i at $node_i"
  until nvidia-smi > /dev/null 2>&1; do
    echo "Waiting for GPU visibility..."
    sleep 2
  done

  srun --nodes=1 --ntasks=1 -w "$node_i" bash -lc '
    IFACE=ib0
    export VLLM_HOST_IP=$(ip -o -4 addr show "$IFACE" | awk "{print \$4}" | cut -d/ -f1)
    if [ -z "$VLLM_HOST_IP" ]; then
      export VLLM_HOST_IP=$(hostname -I | awk "{print \$1}")
    fi
    uv --project "$UV_PROJECT" run ray start --address "'"$MASTER_ADDR:$MASTER_PORT"'" \
      --node-ip-address "$VLLM_HOST_IP" \
      --num-cpus '"$SLURM_CPUS_PER_TASK"' --num-gpus '"$SLURM_GPUS_PER_NODE"' --block
  ' &
  sleep 5
done

sleep 10

uv --project "$UV_PROJECT" run python - << 'EOF'
import os
import time
import ray

address = os.environ["MASTER_ADDR"] + ":" + os.environ["MASTER_PORT"]
deadline = time.time() + int(os.environ.get("PI_VLLM_RAY_READY_TIMEOUT_S", "300"))
last_error = None
while time.time() < deadline:
    try:
        ray.init(address=address)
        print("Cluster resources:", ray.cluster_resources())
        ray.shutdown()
        break
    except Exception as exc:
        last_error = exc
        print(f"Waiting for Ray at {address}: {exc!r}", flush=True)
        time.sleep(5)
else:
    raise RuntimeError(f"Ray did not become ready at {address}") from last_error
EOF

MODEL=${PI_VLLM_MODEL:-/e/data1/datasets/products/mmlaion/shared/models/Qwen/Qwen3.6-27B-FP8/}
SERVED_MODEL_NAME=${PI_VLLM_SERVED_MODEL_NAME:-Qwen3.6-27B-FP8}
VLLM_PORT=${PI_VLLM_PORT:-8080}

srun --overlap --nodes=1 --ntasks=1 -w "$head_node" uv --project "$UV_PROJECT" run vllm serve "$MODEL" \
  --host "$MASTER_IP" --port "$VLLM_PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --enforce-eager \
  --distributed-executor-backend ray \
  --max-model-len 262144 \
  --tensor-parallel-size 1 \
  --data-parallel-size 4 \
  --pipeline-parallel-size 1 \
  --additional-config '{"gdn_prefill_backend":"triton"}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
