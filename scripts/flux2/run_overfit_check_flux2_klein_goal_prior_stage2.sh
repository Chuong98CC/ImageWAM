#!/usr/bin/env bash
# Overfit sanity check: can goal-prior Stage 2 memorise a handful of episodes?
#
# Train on the first N episodes of ONE LIBERO suite, then evaluate on exactly the
# benchmark task those episodes belong to. If the pipeline is sound the policy
# should reach a high success rate on a task it has seen a few times; if it
# cannot fit even that, something is broken and no amount of full-data training
# will fix it. This is the cheap check to run before committing to a long run.
#
# Stage 2 + LoRA only, matching the real run. Stage 1 and full fine-tuning are
# deliberately not supported here; use their own launchers.
#
# Knobs (export before calling):
#   OVERFIT_SUITE      LIBERO suite to overfit (default: libero_spatial)
#   OVERFIT_EPISODES   how many leading episodes (default: 4)
#   PHASE              train | eval | both (default: both)
#   MAX_STEPS          training steps (default: 400)
#   SAVE_EVERY         checkpoint interval (default: 100)
#   NUM_TRIALS         simulator rollouts for the single task (default: 10)
#   EXP_PATH           only for PHASE=eval; the run to evaluate
#   OVERFIT_DRY_RUN    print the commands instead of running them
#
# Usage:
#   bash scripts/flux2/run_overfit_check_flux2_klein_goal_prior_stage2.sh
#   OVERFIT_EPISODES=2 MAX_STEPS=200 bash ...   # smaller still
#   PHASE=eval EXP_PATH=runs/.../2026-09-20_13-00-00 bash ...
#
# Any extra argv is forwarded to the *training* call only, as Hydra overrides:
#   ... log_every=1        # noisy, useful for a first smoke test
# Eval settings are the env knobs above, so a training override cannot leak in.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

# imagewam_init sources .env.local with `set -a`, and .env.local sets DRY_RUN=false,
# so a caller-side DRY_RUN=true never reaches imagewam_run. Read it under a name
# .env.local does not define, otherwise this script cannot be dry-run at all.
DRY_RUN="${OVERFIT_DRY_RUN:-${DRY_RUN:-false}}"
export DRY_RUN

imagewam_require_env DATA_ROOT

OVERFIT_SUITE="${OVERFIT_SUITE:-libero_spatial}"
OVERFIT_EPISODES="${OVERFIT_EPISODES:-4}"
PHASE="${PHASE:-both}"
MAX_STEPS="${MAX_STEPS:-2000}"
SAVE_EVERY="${SAVE_EVERY:-500}"
NUM_TRIALS="${NUM_TRIALS:-10}"

TASK_NAME="libero_flux2_klein_4b_goal_prior_stage2"
RUNS_DIR="${REPO_ROOT}/runs/${TASK_NAME}"
DATASET_DIR="${DATA_ROOT}/${OVERFIT_SUITE}_no_noops_lerobot"
EPISODES_FILE="${DATASET_DIR}/meta/episodes.jsonl"

case "${PHASE}" in
  train|eval|both) ;;
  *)
    echo "PHASE must be one of train|eval|both, got '${PHASE}'" >&2
    exit 2
    ;;
esac

if [ ! -f "${EPISODES_FILE}" ]; then
  echo "No episode metadata at ${EPISODES_FILE}." >&2
  echo "OVERFIT_SUITE='${OVERFIT_SUITE}' must name a suite under DATA_ROOT=${DATA_ROOT}." >&2
  exit 2
fi

EPISODE_COUNT="$(wc -l < "${EPISODES_FILE}" | tr -d ' ')"
if [ "${OVERFIT_EPISODES}" -lt 1 ] || [ "${OVERFIT_EPISODES}" -gt "${EPISODE_COUNT}" ]; then
  echo "OVERFIT_EPISODES=${OVERFIT_EPISODES} must be in [1, ${EPISODE_COUNT}] for ${OVERFIT_SUITE}." >&2
  exit 2
fi

# `episode_index_filter` keeps episodes where `ep % period < keep_first`. Setting
# period to the episode count + 1 makes `ep % period == ep` for every real episode,
# so the rule reduces to exactly `ep < N` -- a clean prefix, with no wrap-around
# phantom episodes. (The existing filter has no explicit episode-list mode; this is
# the exact way to express "the first N" with it.)
PERIOD="$((EPISODE_COUNT + 1))"
EPISODE_FILTER="+data.train.episode_index_filter={mode: periodic_prefix, period: ${PERIOD}, keep_first: ${OVERFIT_EPISODES}}"

imagewam_print_config OVERFIT_SUITE OVERFIT_EPISODES EPISODE_COUNT PHASE MAX_STEPS NUM_TRIALS DATA_ROOT

# `find` exits non-zero on a missing directory, and `set -o pipefail` turns that
# into a script-killing failure -- which is precisely the first-run case, when
# `runs/<task>/` has not been created yet. Hence the trailing `|| true`.
find_dirs() {
  find "$1" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort || true
}

find_new_run_dir() {
  local before="$1" after
  after="$(find_dirs "${RUNS_DIR}")"
  comm -13 <(printf '%s\n' "${before}") <(printf '%s\n' "${after}") | tail -n 1
}

RUN_DIR="${EXP_PATH:-}"

if [ "${PHASE}" = "train" ] || [ "${PHASE}" = "both" ]; then
  echo
  echo "=== Phase 1/2: training on ${OVERFIT_EPISODES} episode(s) of ${OVERFIT_SUITE} ==="
  runs_before="$(find_dirs "${RUNS_DIR}")"

  imagewam_run bash "${SCRIPT_DIR}/run_train_flux2_klein_goal_prior_stage2_lora_1gpu.sh" \
    "data.train.dataset_dirs=[${DATASET_DIR}]" \
    "${EPISODE_FILTER}" \
    "max_steps=${MAX_STEPS}" \
    "save_every=${SAVE_EVERY}" \
    "$@"

  if [ "${DRY_RUN}" = "true" ]; then
    RUN_DIR="${RUN_DIR:-${RUNS_DIR}/<new-run>}"
  else
    RUN_DIR="$(find_new_run_dir "${runs_before}")"
    if [ -z "${RUN_DIR}" ]; then
      echo "Training produced no new run directory under ${RUNS_DIR}." >&2
      exit 1
    fi
  fi
fi

if [ "${PHASE}" = "eval" ]; then
  if [ -z "${RUN_DIR}" ]; then
    echo "PHASE=eval needs EXP_PATH=<run directory>." >&2
    exit 2
  fi
  if [ ! -d "${RUN_DIR}" ]; then
    echo "EXP_PATH does not exist: ${RUN_DIR}" >&2
    exit 2
  fi
fi

echo
echo "=== Phase 2/2: evaluating on the trained task ==="
echo "run: ${RUN_DIR}"

# The dataset's tasks.jsonl index is not the benchmark task_id, so the task file
# is derived from the episode instructions rather than assumed. See the module.
TASK_FILE="${RUN_DIR}/overfit_tasks.txt"
imagewam_run imagewam_python "${SCRIPT_DIR}/overfit_episodes_to_tasks.py" \
  --suite "${OVERFIT_SUITE}" \
  --episodes "${OVERFIT_EPISODES}" \
  --dataset-dir "${DATASET_DIR}" \
  --out "${TASK_FILE}"

# Take the highest step that actually exists rather than assuming max_steps was
# reached -- a run stopped early still has a usable checkpoint.
CKPT_NAME=""
if [ -d "${RUN_DIR}/checkpoints/weights" ]; then
  CKPT_NAME="$(find "${RUN_DIR}/checkpoints/weights" -maxdepth 1 -name 'step_*.pt' -printf '%f\n' | sort | tail -n 1)"
fi
if [ -z "${CKPT_NAME}" ]; then
  if [ "${DRY_RUN}" = "true" ]; then
    CKPT_NAME="step_$(printf '%06d' "${MAX_STEPS}").pt"
  else
    echo "No checkpoint under ${RUN_DIR}/checkpoints/weights." >&2
    exit 1
  fi
fi
EVAL_TRAIN_STEP="${CKPT_NAME#step_}"
EVAL_TRAIN_STEP="${EVAL_TRAIN_STEP%.pt}"
EVAL_TRAIN_STEP="$((10#${EVAL_TRAIN_STEP}))"
echo "checkpoint: ${CKPT_NAME} (step ${EVAL_TRAIN_STEP})"

# The manager re-tags any output_dir under evaluate_results/ as
# <family>/<checkpoint tag>/<run_ts>, so this name reappears one level deeper
# than it is set. The message below globs for where it actually landed.
EVAL_LABEL="overfit_${OVERFIT_SUITE}_ep${OVERFIT_EPISODES}"
EVAL_OUT_DIR="${REPO_ROOT}/evaluate_results/libero/${EVAL_LABEL}"

TASK="${TASK_NAME}" \
EXP_PATH="${RUN_DIR}" \
EVAL_TRAIN_STEP="${EVAL_TRAIN_STEP}" \
NUM_GPUS="${EVAL_NUM_GPUS:-1}" \
MAX_TASKS_PER_GPU=1 \
NUM_TRIALS="${NUM_TRIALS}" \
TASK_SUITE_NAMES="[${OVERFIT_SUITE}]" \
  imagewam_run bash "${SCRIPT_DIR}/run_eval_flux2_libero.sh" \
    "MULTIRUN.task_file=${TASK_FILE}" \
    "EVALUATION.output_dir=${EVAL_OUT_DIR}"

if [ "${DRY_RUN}" != "true" ]; then
  echo
  echo "Results (summary.json, task_success_rates.csv) under:"
  find "${REPO_ROOT}/evaluate_results" -mindepth 2 -maxdepth 3 -type d -name "${EVAL_LABEL}" 2>/dev/null || true
fi
