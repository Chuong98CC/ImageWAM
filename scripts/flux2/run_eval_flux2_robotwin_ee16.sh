#!/usr/bin/env bash
set -euo pipefail

# Evaluate a RoboTwin checkpoint fine-tuned in the 16D pretrain space. The
# policy reverses the processor transform before sending 14D actions to RoboTwin.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."
# See run_eval_flux2_robotwin.sh: RoboTwin eval runs from the dedicated env.
imagewam_robotwin_env

CONFIG_NAME="sim_robotwin"
TASK="${EE16_TASK:-robotwin_flux2_klein_4b_base_clean_imagewam_ee16}"

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
LIBERO_WORKER_ENV_SOURCE="${LIBERO_WORKER_ENV_SOURCE:-}"
export LIBERO_WORKER_ENV_SOURCE

imagewam_prepare_eval_ckpt

COMMON=(
  --config-name "${CONFIG_NAME}"
  task="${TASK}"
  ckpt="${CKPT_PATH}"
  EVALUATION.dataset_stats_path="${DATASET_STATS_PATH}"
  model.flux2_src_path="${FLUX2_SRC}"
  model.flux2_model_path="${FLUX2_MODEL_PATH}"
  model.ae_model_path="${FLUX2_AE_MODEL_PATH}"
  model.variant=klein-base-4b
  model.qwen3_model_spec="${FLUX2_QWEN3_MODEL_SPEC}"
  model.qwen_context_len=128
  model.load_text_encoder=true
  model.proprio_dim=16
  model.pack_proprio_after_text=true
  MULTIRUN.num_gpus="${NUM_GPUS:-8}"
  MULTIRUN.max_tasks_per_gpu="${MAX_TASKS_PER_GPU:-3}"
  MULTIRUN.phases="${PHASES:-[clean,random]}"
  EVALUATION.eval_num_episodes="${EVAL_NUM_EPISODES:-50}"
  EVALUATION.action_horizon="${ACTION_HORIZON:-16}"
  EVALUATION.replan_steps="${REPLAN_STEPS:-16}"
  EVALUATION.skip_get_obs_within_replan="${SKIP_GET_OBS_WITHIN_REPLAN:-true}"
  EVALUATION.robotwin_camera_layout="${ROBOTWIN_CAMERA_LAYOUT:-compact_288x256}"
  EVALUATION.apply_action_transform_backward="${APPLY_ACTION_TRANSFORM_BACKWARD:-true}"
)

if [ -n "${QWEN_CACHE_DIR}" ]; then
  COMMON+=(data.train.qwen_text_cache_dir="${QWEN_CACHE_DIR}")
fi

imagewam_print_config \
  TASK CKPT_PATH DATASET_STATS_PATH FLUX2_SRC FLUX2_MODEL_PATH \
  FLUX2_AE_MODEL_PATH FLUX2_QWEN3_MODEL_SPEC ROBOTWIN_VENV
imagewam_run imagewam_python experiments/robotwin/run_robotwin_manager.py "${COMMON[@]}" "$@"
