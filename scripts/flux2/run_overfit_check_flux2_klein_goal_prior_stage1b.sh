#!/usr/bin/env bash
# Stage 1b overfit sanity check: can the aggregator memorise a handful of SE(3) goals?
#
# Train on the first N episodes of ONE LIBERO suite, then read the validation
# `loss_pose` / `loss_video` at the end of the run. If the pipeline is sound both
# should fall toward zero on data the model has seen a few times; if they plateau,
# something is broken and no amount of full-data training will fix it.
#
# There is no rollout here, and there cannot be one. Stage 1b excludes the Action
# Expert, so the model emits no actions and LIBERO's success rate is undefined for
# it. The overfit signal is the loss curve, read off validation -- which is the
# *training* episodes, because `configs/data/libero_omnigen2_pair.yaml` defines no
# `val` split and `build_datasets` then sets `val_ds = train_ds`.
#
# Compare against `run_overfit_check_flux2_klein_goal_prior_stage2.sh`, which does
# the same check for the stage that has a policy, and evaluates it in the simulator.
#
# Stage 1b + LoRA only. Stage 1 and full fine-tuning are deliberately not supported
# here; use their own launchers.
#
# Knobs (export before calling):
#   OVERFIT_SUITE      LIBERO suite to overfit (default: libero_spatial)
#   OVERFIT_EPISODES   how many leading episodes (default: 4)
#   MAX_STEPS          training steps (default: 400)
#   SAVE_EVERY         checkpoint interval (default: 100)
#   EVAL_EVERY         validation interval (default: 100; 0 disables)
#   OVERFIT_DRY_RUN    print the command instead of running it
#
# Usage:
#   bash scripts/flux2/run_overfit_check_flux2_klein_goal_prior_stage1b.sh
#   OVERFIT_EPISODES=2 MAX_STEPS=200 bash ...   # smaller still
#
# Any extra argv is forwarded to the training call as Hydra overrides:
#   ... log_every=1        # noisy, useful for a first smoke test
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
MAX_STEPS="${MAX_STEPS:-400}"
SAVE_EVERY="${SAVE_EVERY:-100}"
EVAL_EVERY="${EVAL_EVERY:-100}"

TASK_NAME="libero_flux2_klein_4b_goal_prior_stage1b"
RUNS_DIR="${REPO_ROOT}/runs/${TASK_NAME}"
DATASET_DIR="${DATA_ROOT}/${OVERFIT_SUITE}_no_noops_lerobot"
EPISODES_FILE="${DATASET_DIR}/meta/episodes.jsonl"

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
if [ "${EVAL_EVERY}" -gt 0 ] && [ "${EVAL_EVERY}" -gt "${MAX_STEPS}" ]; then
  echo "EVAL_EVERY=${EVAL_EVERY} exceeds MAX_STEPS=${MAX_STEPS}, so the run would" >&2
  echo "finish before it ever validates and report no loss_pose at all." >&2
  exit 2
fi

# `episode_index_filter` keeps episodes where `ep % period < keep_first`. Setting
# period to the episode count + 1 makes `ep % period == ep` for every real episode,
# so the rule reduces to exactly `ep < N` -- a clean prefix, with no wrap-around
# phantom episodes. (The existing filter has no explicit episode-list mode; this is
# the exact way to express "the first N" with it.)
PERIOD="$((EPISODE_COUNT + 1))"
EPISODE_FILTER="+data.train.episode_index_filter={mode: periodic_prefix, period: ${PERIOD}, keep_first: ${OVERFIT_EPISODES}}"

imagewam_print_config OVERFIT_SUITE OVERFIT_EPISODES EPISODE_COUNT MAX_STEPS EVAL_EVERY DATA_ROOT

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

echo
echo "=== Training on ${OVERFIT_EPISODES} episode(s) of ${OVERFIT_SUITE} ==="
runs_before="$(find_dirs "${RUNS_DIR}")"

TRAIN_ARGS=(
  "data.train.dataset_dirs=[${DATASET_DIR}]"
  "${EPISODE_FILTER}"
  "max_steps=${MAX_STEPS}"
  "save_every=${SAVE_EVERY}"
  "eval_every=${EVAL_EVERY}"
  "$@"
)

mkdir -p "${RUNS_DIR}"
RUN_LOG="${RUNS_DIR}/overfit_${OVERFIT_SUITE}_ep${OVERFIT_EPISODES}.log"

if [ "${DRY_RUN}" = "true" ]; then
  imagewam_run bash "${SCRIPT_DIR}/run_train_flux2_klein_goal_prior_stage1b_lora_1gpu.sh" "${TRAIN_ARGS[@]}"
  RUN_DIR="${RUNS_DIR}/<new-run>"
else
  # Tee'd so the validation lines survive the run: `loss_pose` is the whole
  # result of this check and the trainer only prints it.
  imagewam_run bash "${SCRIPT_DIR}/run_train_flux2_klein_goal_prior_stage1b_lora_1gpu.sh" \
    "${TRAIN_ARGS[@]}" 2>&1 | tee "${RUN_LOG}"

  RUN_DIR="$(find_new_run_dir "${runs_before}")"
  if [ -z "${RUN_DIR}" ]; then
    echo "Training produced no new run directory under ${RUNS_DIR}." >&2
    exit 1
  fi
fi

echo
echo "=== Result ==="
echo "run:  ${RUN_DIR}"
echo "log:  ${RUN_LOG}"

if [ "${DRY_RUN}" = "true" ]; then
  exit 0
fi

# The last validation lines carry the number this check exists to produce. Read
# them rather than diffing the first against the last by hand.
VALIDATION_LINES="$(grep -E 'loss_pose=' "${RUN_LOG}" 2>/dev/null | tail -n 4 || true)"
if [ -z "${VALIDATION_LINES}" ]; then
  echo
  echo "No validation lines with loss_pose found. Either EVAL_EVERY=0, or the run" >&2
  echo "stopped before its first validation -- check ${RUN_LOG}." >&2
  exit 1
fi
echo
echo "last validations (loss_pose and loss_video should be falling):"
printf '%s\n' "${VALIDATION_LINES}"
echo
echo "checkpoints under: ${RUN_DIR}/checkpoints/weights"
