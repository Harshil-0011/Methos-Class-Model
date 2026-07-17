#!/usr/bin/env bash
# =============================================================================
# 4-GPU FSDP Training — NSLT on 4x A100 80GB (320GB pooled)
# =============================================================================
# Usage:
#   bash scripts/train_4gpu.sh                        # start full training
#   bash scripts/train_4gpu.sh --fresh-start           # ignore checkpoints
#   bash scripts/train_4gpu.sh config-validate         # validate config
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${CONFIG:-$SCRIPT_DIR/config.yaml}"
# Auto-detect GPU count from CUDA_VISIBLE_DEVICES if set
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=${#GPU_LIST[@]}
else
    NUM_GPUS=${NUM_GPUS:-4}
fi
MASTER_PORT=${MASTER_PORT:-29500}

if [ "${1:-}" = "" ] || [[ "${1:-}" == --* ]]; then
    CMD="full-training"
else
    CMD="$1"
    shift
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS=8
# Force NCCL settings (don't use fallback — override container defaults)
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "====================================================================="
echo "  Launching 4-GPU FSDP training"
echo "  Config:   ${CONFIG}"
echo "  GPUs:     ${NUM_GPUS} x A100-80GB (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3})"
echo "  Strategy: FSDP full_shard"
echo "  Command:  ${CMD}"
echo "====================================================================="

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/main.py" \
    --config "${CONFIG}" \
    "${CMD}" \
    "$@"
