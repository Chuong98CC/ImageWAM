#!/usr/bin/env bash
# Stage 2 + LoRA on a single 32GB GPU (RTX 5090).
#
# `run_train_flux2_klein_goal_prior_stage2.sh` with the settings that differ when
# you have one card instead of eight. Everything else -- the LoRA plumbing, the
# merge-on-save rule, the Stage1->Stage2 bridge -- is unchanged; read that script
# and docs/flux2_architecture.md §5.2 for the mechanism.
#
# Why each override exists (export any of them before the call to change it):
#
#   GPU_PER_NODE=1   the parent defaults to 8, which on one card launches 8 ranks
#                    that each load the 4B expert. That is an instant OOM.
#   TRAIN_NORM_STATS the parent defaults to a baseline run directory that does not
#                    exist here. The released Stage 1 dataset_stats.json is
#                    numerically identical (1712 episodes, 277713 transitions),
#                    which is what makes stage 1 and stage 2 comparable.
#   QWEN_CACHE_DIR   no Qwen3 text cache ships with the repo. Without one, every
#                    sample tries to embed on the fly and fails, because
#                    `load_text_encoder: false` means no text encoder is loaded.
#                    The parent precomputes it on first run (~10 min, 301MB).
#   batch_size=2     NOT a LoRA saving. LoRA removes optimizer state (46GB ->
#                    0.19GB) but every activation still has to exist, because the
#                    aggregator needs gradient through all 25 frozen layers.
#                    2 frames x 256 tokens + 512 text = 1024 tokens/sample, and at
#                    batch 10 the single blocks' linear1 tensor alone is 4.2GB.
#   gradient_accumulation_steps=5   keeps the effective batch at 10, matching the
#                    evaluated run's optimizer geometry.
#   eval_every=0     validation renders decoded video; a second 8-sample forward
#                    plus a decode is the likeliest OOM here. Turn it back on once
#                    a training step is known to fit.
#
# Requires DATA_ROOT to point at the LIBERO tree (`$LIBERO_ROOT` in .env.local),
# and the Qwen cache is written under it.
#
# Usage:
#   bash scripts/flux2/run_train_flux2_klein_goal_prior_stage2_lora_1gpu.sh
#   ... max_steps=20 save_every=20 log_every=1      # smoke test first
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

export GPU_PER_NODE="${GPU_PER_NODE:-1}"
export FLUX2_LORA_ENABLED="${FLUX2_LORA_ENABLED:-true}"
export QWEN_CACHE_BATCH_SIZE="${QWEN_CACHE_BATCH_SIZE:-4}"
export QWEN_CACHE_DIR="${QWEN_CACHE_DIR:-${DATA_ROOT}/flux2_qwen3_cache_${FLUX2_VARIANT:-4b}}"
export TRAIN_NORM_STATS="${TRAIN_NORM_STATS:-${REPO_ROOT}/LIT_ckpt/imagewam/lit_stage1/dataset_stats.json}"
export STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${REPO_ROOT}/LIT_ckpt/imagewam/lit_stage1/model.pt}"

if [ ! -f "${STAGE1_CHECKPOINT}" ]; then
  echo "Missing Stage1 checkpoint: ${STAGE1_CHECKPOINT}" >&2
  exit 2
fi
if [ ! -f "${TRAIN_NORM_STATS}" ]; then
  echo "Missing dataset stats: ${TRAIN_NORM_STATS}" >&2
  exit 2
fi
case "${DATA_ROOT}" in
  */libero_*) ;;
  *)
    echo "DATA_ROOT=${DATA_ROOT} does not look like a LIBERO tree." >&2
    echo "Stage 2 trains on LIBERO; set DATA_ROOT=\$LIBERO_ROOT in .env.local." >&2
    exit 2
    ;;
esac

imagewam_print_config GPU_PER_NODE DATA_ROOT QWEN_CACHE_DIR STAGE1_CHECKPOINT TRAIN_NORM_STATS FLUX2_LORA_ENABLED

imagewam_run bash "${SCRIPT_DIR}/run_train_flux2_klein_goal_prior_stage2.sh" \
  batch_size="${BATCH_SIZE:-2}" \
  num_workers="${NUM_WORKERS:-6}" \
  prefetch_factor="${PREFETCH_FACTOR:-2}" \
  persistent_workers="${PERSISTENT_WORKERS:-true}" \
  gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-5}" \
  eval_every="${EVAL_EVERY:-0}" \
  "$@"
