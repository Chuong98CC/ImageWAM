#!/usr/bin/env bash
# Stage 1b + LoRA on a single 32GB GPU (RTX 5090).
#
# `run_train_flux2_klein_goal_prior_stage2.sh` with GOAL_PRIOR_STAGE=stage1b and
# the settings that differ when you have one card instead of eight. Read that
# script's header for what Stage 1b is; read docs/flux2_architecture.md §5.2 for
# the LoRA mechanism.
#
# What is *not* needed here, unlike the Stage 2 wrapper:
#
#   STAGE1_CHECKPOINT  Stage 1's two products are `goal_pose_encoder`, which the
#                      Stage1->2 bridge drops, and the Action Expert, which Stage
#                      1b never runs. There is nothing to inherit, so Stage 1b
#                      trains from the base FLUX.2 init with a fresh aggregator.
#
# What is still needed, and why:
#
#   GPU_PER_NODE=1   the parent defaults to 8, which on one card launches 8 ranks
#                    that each load the 4B expert. That is an instant OOM.
#   TRAIN_NORM_STATS the released Stage 1 dataset_stats.json is numerically
#                    identical to the baseline's (1712 episodes, 277713
#                    transitions), which is what keeps the min/max -- and so the
#                    video and pose losses -- comparable across runs.
#   QWEN_CACHE_DIR   no Qwen3 text cache ships with the repo. Without one every
#                    sample tries to embed on the fly and fails, because
#                    `load_text_encoder: false` means no text encoder is loaded.
#   batch_size=2     NOT a LoRA saving. LoRA removes optimizer state but every
#                    activation still has to exist, because the aggregator needs
#                    gradient through all 25 layers it reads.
#   gradient_accumulation_steps=5   keeps the effective batch at 10.
#   eval_every=0     validation is cheap here (Stage 1b skips the rollout and the
#                    VAE decode entirely, so no video is rendered), but the first
#                    forward is not free. Turn it on once a step is known to fit.
#
# Usage:
#   bash scripts/flux2/run_train_flux2_klein_goal_prior_stage1b_lora_1gpu.sh
#   ... max_steps=20 save_every=20 eval_every=20 log_every=1   # smoke test
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

export GOAL_PRIOR_STAGE="stage1b"
export GPU_PER_NODE="${GPU_PER_NODE:-1}"
export FLUX2_LORA_ENABLED="${FLUX2_LORA_ENABLED:-true}"
export QWEN_CACHE_BATCH_SIZE="${QWEN_CACHE_BATCH_SIZE:-4}"
export QWEN_CACHE_DIR="${QWEN_CACHE_DIR:-${DATA_ROOT}/flux2_qwen3_cache_${FLUX2_VARIANT:-4b}}"
export TRAIN_NORM_STATS="${TRAIN_NORM_STATS:-${REPO_ROOT}/LIT_ckpt/imagewam/lit_stage1/dataset_stats.json}"

if [ ! -f "${TRAIN_NORM_STATS}" ]; then
  echo "Missing dataset stats: ${TRAIN_NORM_STATS}" >&2
  exit 2
fi
case "${DATA_ROOT}" in
  */libero_*) ;;
  *)
    echo "DATA_ROOT=${DATA_ROOT} does not look like a LIBERO tree." >&2
    echo "Stage 1b trains on LIBERO; set DATA_ROOT=\$LIBERO_ROOT in .env.local." >&2
    exit 2
    ;;
esac

imagewam_print_config GOAL_PRIOR_STAGE GPU_PER_NODE DATA_ROOT QWEN_CACHE_DIR TRAIN_NORM_STATS FLUX2_LORA_ENABLED

imagewam_run bash "${SCRIPT_DIR}/run_train_flux2_klein_goal_prior_stage2.sh" \
  batch_size="${BATCH_SIZE:-2}" \
  num_workers="${NUM_WORKERS:-6}" \
  prefetch_factor="${PREFETCH_FACTOR:-2}" \
  persistent_workers="${PERSISTENT_WORKERS:-true}" \
  gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-5}" \
  eval_every="${EVAL_EVERY:-0}" \
  "$@"
