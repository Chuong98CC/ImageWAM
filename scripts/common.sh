#!/usr/bin/env bash
# Shared helpers for ImageWAM release scripts.
# Source this file from bash entrypoints after `set -euo pipefail`.

imagewam_init() {
  local default_root="$1"
  REPO_ROOT="${REPO_ROOT:-$(cd "${default_root}" && pwd)}"
  export REPO_ROOT
  cd "${REPO_ROOT}"

  if [ -f "${REPO_ROOT}/.env.local" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env.local"
    set +a
  fi

  # TorchCodec's dlopen wants CUDA NPP: libtorchcodec_core*.so links libnppicc.so.12,
  # which presently resolves from the system CUDA toolkit via ldconfig -- no venv here
  # ships an nvidia/npp wheel, so this loop is normally a no-op. Probe both venvs so it
  # keeps working if a lock change ever adds one (RoboTwin eval runs from .venv_rb2).
  local _venv _npp_lib
  for _venv in "${REPO_ROOT}/.venv" "${ROBOTWIN_VENV:-${REPO_ROOT}/.venv_rb2}"; do
    for _npp_lib in "${_venv}"/lib/python*/site-packages/nvidia/npp/lib; do
      if [ -d "${_npp_lib}" ]; then
        case ":${LD_LIBRARY_PATH:-}:" in
          *":${_npp_lib}:"*) ;;
          *) export LD_LIBRARY_PATH="${_npp_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
        esac
      fi
    done
  done
}

imagewam_require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "Missing required environment variable: ${name}" >&2
    echo "Set it in the shell or in ${REPO_ROOT}/.env.local" >&2
    exit 2
  fi
}

imagewam_print_config() {
  if [ "${IMAGEWAM_QUIET:-false}" = "true" ]; then
    return 0
  fi
  local name
  for name in "$@"; do
    printf '[config] %s=%s\n' "${name}" "${!name:-<unset>}"
  done
}

imagewam_run() {
  if [ "${DRY_RUN:-false}" = "true" ]; then
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

imagewam_activate_env() {
  local env_file="${1:-}"
  if [ -n "${env_file}" ]; then
    # shellcheck disable=SC1090
    source "${env_file}"
  fi
}

imagewam_python() {
  "${PYTHON_BIN:-python}" "$@"
}

# Switch to the RoboTwin evaluation venv (built by scripts/setup/install_robotwin_env.sh).
# Activation is load-bearing, not cosmetic: `uv pip` inside RoboTwin's own scripts takes its
# target from VIRTUAL_ENV, and without it uv walks up from third_party/RoboTwin and finds the
# main .venv -- which is what used to rewrite .venv's pinned versions.
imagewam_robotwin_env() {
  local venv="${ROBOTWIN_VENV:-${REPO_ROOT}/.venv_rb2}"
  if [ ! -x "${venv}/bin/python" ]; then
    echo "RoboTwin venv not found at ${venv}" >&2
    echo "Run scripts/setup/install_robotwin_env.sh, or point ROBOTWIN_VENV at one." >&2
    exit 2
  fi
  # shellcheck disable=SC1091
  source "${venv}/bin/activate"
  PYTHON_BIN="${venv}/bin/python"
  export PYTHON_BIN
}

imagewam_ckpt_from_exp() {
  if [ -z "${CKPT_PATH:-}" ]; then
    imagewam_require_env EXP_PATH
    imagewam_require_env EVAL_TRAIN_STEP
    CKPT_PATH="${EXP_PATH}/checkpoints/weights/step_${EVAL_TRAIN_STEP}.pt"
    export CKPT_PATH
  fi
  if [ -z "${DATASET_STATS_PATH:-}" ] && [ -n "${EXP_PATH:-}" ]; then
    DATASET_STATS_PATH="${EXP_PATH}/dataset_stats.json"
    export DATASET_STATS_PATH
  fi
}

imagewam_prepare_eval_ckpt() {
  if [ -n "${LOCAL_CKPT_ROOT:-}" ]; then
    imagewam_require_env CKPT_PATH
    if [ "${DRY_RUN:-false}" = "true" ]; then
      echo "[dry-run] skipping local checkpoint copy"
      return 0
    fi
    local task_name="${TASK:-eval}"
    local run_name="$(basename "$(dirname "$(dirname "$(dirname "${CKPT_PATH}")")")")"
    local local_path="${LOCAL_CKPT_ROOT}/runs/${task_name}/${run_name}/checkpoints/weights/$(basename "${CKPT_PATH}")"
    mkdir -p "$(dirname "${local_path}")"
    if [ ! -f "${local_path}" ] || [ "${CKPT_PATH}" -nt "${local_path}" ]; then
      echo "Copying checkpoint to local disk: ${local_path}"
      cp "${CKPT_PATH}" "${local_path}.tmp"
      mv "${local_path}.tmp" "${local_path}"
    else
      echo "Using existing local checkpoint: ${local_path}"
    fi
    CKPT_PATH="${local_path}"
    export CKPT_PATH
  fi
}
