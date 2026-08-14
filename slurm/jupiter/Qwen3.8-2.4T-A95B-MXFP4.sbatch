#!/bin/bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=10:00:00
#SBATCH --job-name=vllm-kimi-k3

set -euo pipefail
set -x
NET_IFACE=ib0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-3600}
export HF_HUB_OFFLINE=1
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
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
export VLLM_USE_RAY_COMPILED_DAG=0
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
APPTAINER_IMAGE=/e/data1/datasets/playground/mmlaion/shared/cherti1/apptainer_images/vllm_qwen3.8.sif
RUN="apptainer exec --nv $APPTAINER_IMAGE"

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
    '"$RUN"' ray start --head --node-ip-address="$VLLM_HOST_IP" \
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
    '"$RUN"' ray start --address "'"$MASTER_ADDR:$MASTER_PORT"'" \
      --node-ip-address "$VLLM_HOST_IP" \
      --num-cpus '"$SLURM_CPUS_PER_TASK"' --num-gpus '"$SLURM_GPUS_PER_NODE"' --block
  ' &
  sleep 5
done

sleep 10

$RUN python3 - << 'EOF'
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

MODEL=/e/data1/datasets/products/mmlaion/shared/models/Qwen/Qwen3.8-2.4T-A95B-MXFP4

SERVED_MODEL_NAME=Qwen3.8-2.4T-A95B-MXFP4
VLLM_PORT=8080
srun --overlap --nodes=1 --ntasks=1 -w "$head_node" $RUN vllm serve "$MODEL" \
  --host "$MASTER_IP" --port "$VLLM_PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --trust-remote-code --gpu-memory-utilization 0.7 \
  --disable-custom-all-reduce \
  --distributed-executor-backend ray \
  --max-model-len 262144 \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 8 \
  --load-format fastsafetensors \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
