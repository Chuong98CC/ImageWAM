#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

# Stock LIBERO, which lives in its own venv: LIBERO and LIBERO-plus both install a package
# named "libero", so sharing a venv means reinstalling one over the other on every switch. The
# manager (it enumerates the task list) and the workers must use the same venv, and
# LIBERO_CONFIG_PATH must point at that benchmark's assets -- so all three are derived here
# rather than left to .env.local.
LIBERO_VENV="${LIBERO_VENV:-${REPO_ROOT}/.venv}"
if [ ! -x "${LIBERO_VENV}/bin/python" ]; then
  echo "LIBERO venv not found: ${LIBERO_VENV}" >&2
  echo "Install it with: bash scripts/setup/_install_libero_env.sh" >&2
  exit 2
fi
LIBERO_PKG_DIR="$("${LIBERO_VENV}/bin/python" -c 'import libero, os; print(os.path.dirname(libero.__file__))' 2>/dev/null || echo "<no libero installed>")"
case "${LIBERO_PKG_DIR}" in
  */LIBERO-plus/*)
    echo "${LIBERO_VENV} has LIBERO-plus installed, but this launcher evaluates stock LIBERO." >&2
    echo "Use scripts/flux2/run_eval_flux2_libero_plus.sh, or install this benchmark first:" >&2
    echo "  bash scripts/setup/_install_libero_env.sh" >&2
    exit 2
    ;;
  "<no libero installed>")
    echo "No libero package in ${LIBERO_VENV}. Install it with:" >&2
    echo "  bash scripts/setup/_install_libero_env.sh" >&2
    exit 2
    ;;
esac
PYTHON_BIN="${LIBERO_VENV}/bin/python"
LIBERO_WORKER_ENV_SOURCE="${LIBERO_VENV}/bin/activate"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${HOME}/.libero}"
export PYTHON_BIN LIBERO_WORKER_ENV_SOURCE LIBERO_CONFIG_PATH

SUITE="libero"
CONFIG_NAME="sim_libero_omnigen2"
FLUX2_VARIANT="${FLUX2_VARIANT:-4b}" # 4b | 9b
TASK="${TASK:-libero_flux2_klein_${FLUX2_VARIANT}_base_imagewam}"
if [ "false" = "true" ]; then
  TASK="${TASK/_imagewam/_clean_imagewam}"
fi

imagewam_require_env FLUX2_SRC
imagewam_require_env FLUX2_AE_MODEL_PATH
imagewam_require_env FLUX2_MODEL_PATH
imagewam_ckpt_from_exp
imagewam_require_env CKPT_PATH
imagewam_require_env DATASET_STATS_PATH

FLUX2_QWEN3_MODEL_SPEC="${FLUX2_QWEN3_MODEL_SPEC:-Qwen/Qwen3-4B}"
QWEN_CACHE_DIR="${QWEN_CACHE_DIR:-}"
export PYTHONPATH="${REPO_ROOT}/src:${FLUX2_SRC}/src:${FLUX2_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
export WORKER_PYTHONPATH="${PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
imagewam_prepare_eval_ckpt

# Resume an interrupted run instead of evaluating everything again. Pass the directory the
# previous run printed ("Evaluation results will be saved to: ..."), or set RESUME=1 to pick
# the newest resumable run for this checkpoint:
#   RESUME_DIR=evaluate_results/libero_omnigen2/model/20260923_201444 bash scripts/flux2/run_eval_flux2_libero.sh
#   RESUME=1 bash scripts/flux2/run_eval_flux2_libero.sh
RESUME="${RESUME:-false}"
RESUME_DIR="${RESUME_DIR:-}"
RESUME_FROM=""
case "${RESUME}" in
  1 | [Tt]rue | [Yy]es) RESUME_FROM="latest" ;;
esac
if [ -n "${RESUME_DIR}" ]; then
  if [ ! -d "${RESUME_DIR}" ]; then
    echo "RESUME_DIR is not a directory: ${RESUME_DIR}" >&2
    exit 2
  fi
  RESUME_FROM="${RESUME_DIR}" # an explicit directory wins over RESUME=1
fi

COMMON=(
  --config-name "${CONFIG_NAME}"
  task="${TASK}"
  ckpt="${CKPT_PATH}"
  EVALUATION.dataset_stats_path="${DATASET_STATS_PATH}"
  model.flux2_src_path="${FLUX2_SRC}"
  model.flux2_model_path="${FLUX2_MODEL_PATH}"
  model.ae_model_path="${FLUX2_AE_MODEL_PATH}"
  model.variant="klein-base-${FLUX2_VARIANT}"
  model.qwen3_model_spec="${FLUX2_QWEN3_MODEL_SPEC}"
  model.load_text_encoder=true
  model.pack_proprio_after_text=true
  MULTIRUN.num_gpus="${NUM_GPUS:-8}"
  MULTIRUN.max_tasks_per_gpu="${MAX_TASKS_PER_GPU:-2}"
  EVALUATION.action_horizon="${ACTION_HORIZON:-16}"
  EVALUATION.replan_steps="${REPLAN_STEPS:-12}"
  EVALUATION.save_rollout_video="${SAVE_ROLLOUT_VIDEO:-false}"
)

if [ -n "${QWEN_CACHE_DIR}" ]; then
  COMMON+=(data.train.qwen_text_cache_dir="${QWEN_CACHE_DIR}")
fi

if [ -n "${RESUME_FROM}" ]; then
  COMMON+=(MULTIRUN.resume_from="${RESUME_FROM}")
fi

COMMON+=(
  model.proprio_dim="${PROPRIO_DIM:-8}"
  data.train.qwen_context_len="${QWEN_CONTEXT_LEN:-512}"
  data.train.qwen_text_cache_format=qwen3_flux2
  MULTIRUN.task_suite_names="${TASK_SUITE_NAMES:-[libero_10,libero_goal,libero_spatial,libero_object]}"
  EVALUATION.num_trials="${NUM_TRIALS:-25}"
  MULTIRUN.chunk_size="${CHUNK_SIZE:-1}"
)

imagewam_print_config SUITE TASK CKPT_PATH DATASET_STATS_PATH FLUX2_SRC FLUX2_MODEL_PATH FLUX2_AE_MODEL_PATH
if [ -n "${RESUME_FROM}" ]; then
  echo "[config] RESUME_FROM=${RESUME_FROM}"
fi
imagewam_run imagewam_python experiments/libero/run_libero_manager.py "${COMMON[@]}" "$@"
