#!/bin/bash
#SBATCH --job-name=quantweave-train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=quantweave-%j.log
#
# Multi-node expert/tensor-parallel training via torchrun, one Slurm task per node (srun), each task then
# spawning --nproc-per-node local processes — torchrun's own multi-node convention, not something Slurm-specific.
#
# NOT verified on a real multi-node cluster: this repository's own testing has one GPU and no multi-node hardware
# (see FUTURE_IDEAS.md and README's "Current Scope"), so this is a documented starting point, not a validated one.
# Check your cluster's rendezvous networking (the c10d backend below needs the head node reachable by hostname
# from every other node on MASTER_PORT) and NCCL settings (NCCL_SOCKET_IFNAME, NCCL_IB_HCA, etc. — cluster-
# specific) before trusting it with a long run.
#
# Submit with, e.g.:
#   sbatch --nodes=2 --gpus-per-node=4 \
#     --export=MOE_TOTAL_EXPERTS=32,MOE_DATA=data/smoke/tinystories.jsonl \
#     deploy/slurm/train_multi_node.sbatch

set -euo pipefail

NNODES="${SLURM_NNODES}"
GPUS_PER_NODE="${MOE_GPUS_PER_NODE:-4}"
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
MASTER_PORT="${MOE_MASTER_PORT:-29500}"

cd "${SLURM_SUBMIT_DIR}"

srun --ntasks-per-node=1 \
  "${MOE_TORCHRUN:-.axolotl-venv/bin/torchrun}" \
    --nnodes "${NNODES}" \
    --nproc_per_node "${GPUS_PER_NODE}" \
    --rdzv_id "${SLURM_JOB_ID}" \
    --rdzv_backend c10d \
    --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
    src/train_quantweave_moe.py \
      --expert-parallel \
      --tensor-parallel "${MOE_TENSOR_PARALLEL:-1}" \
      --data "${MOE_DATA:-data/smoke/tinystories.jsonl}" \
      --output "${MOE_OUTPUT:-artifacts/outputs/quantweave-moe-ep-out}" \
      --steps "${MOE_STEPS:-10000}" \
      --examples "${MOE_EXAMPLES:-100000}" \
      --sequence-length "${MOE_SEQUENCE_LENGTH:-128}" \
      --batch-size "${MOE_BATCH_SIZE:-2}" \
      --total-experts "${MOE_TOTAL_EXPERTS:-32}" \
      --active-experts "${MOE_ACTIVE_EXPERTS:-2}" \
      --device cuda \
      --checkpoint-dir "${MOE_CHECKPOINT_DIR:-artifacts/checkpoints/quantweave-moe-ep-checkpoint}" \
      --checkpoint-interval "${MOE_CHECKPOINT_INTERVAL:-1000}" \
      ${MOE_EXTRA_ARGS:-}
