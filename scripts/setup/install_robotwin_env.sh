#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."
ROBOTWIN_DIR="${ROBOTWIN_DIR:-${REPO_ROOT}/third_party/RoboTwin}"
# RoboTwin's requirements_mod.txt is largely unpinned (moviepy caps pillow<12, plus torch,
# torchvision, huggingface_hub, av, wandb, scipy, matplotlib), so installing it into .venv
# downgrades pins that pyproject.toml/uv.lock guarantee. Keep it in a second env; .venv stays
# exactly as `uv sync` built it, for LIBERO / FLUX.2 / OmniGen2.
ROBOTWIN_VENV="${ROBOTWIN_VENV:-${REPO_ROOT}/.venv_rb2}"
# FLUX.2 needs 4.56.1; the lock pins 4.51.3 (see pyproject.toml).
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.56.1}"

############# USE THIS IF YOUR GPU IS IN DOCKER ENV AND NOT WITH NVIDIA_DRIVER_CAPABILITIES ENV SET TO GRAPHICS

# IMAGEWAM_TMP_DIR="${IMAGEWAM_TMP_DIR:-${REPO_ROOT}/.tmp}"
# FAKE_KMOD_DIR="${FAKE_KMOD_DIR:-${IMAGEWAM_TMP_DIR}/fake-kmod}"
# NVIDIA_RUNFILE="${NVIDIA_RUNFILE:-}"
# VULKAN_ICD_PATH="${VULKAN_ICD_PATH:-${IMAGEWAM_TMP_DIR}/nvidia_icd.json}"


# mkdir -p "${FAKE_KMOD_DIR}" "$(dirname "${VULKAN_ICD_PATH}")"
# for cmd in modprobe rmmod insmod lsmod depmod; do
#   cat > "${FAKE_KMOD_DIR}/${cmd}" <<'EOF'
# #!/usr/bin/env bash
# echo "[fake kmod] $(basename "$0") $@" >&2
# exit 0
# EOF
#   chmod +x "${FAKE_KMOD_DIR}/${cmd}"
# done
# export PATH="${FAKE_KMOD_DIR}:$PATH"

# if [ -n "${NVIDIA_RUNFILE}" ]; then
#   imagewam_run sh "${NVIDIA_RUNFILE}" --accept-license --no-questions --ui=none --no-kernel-module --no-drm --no-nvidia-modprobe
# fi

# USE LINES 
# cat > "${VULKAN_ICD_PATH}" <<'EOF'
# {
#   "file_format_version": "1.0.0",
#   "ICD": {
#     "library_path": "/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0",
#     "api_version": "1.3.0"
#   }
# }
# EOF
# export VK_ICD_FILENAMES="${VULKAN_ICD_PATH}"
# export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

# imagewam_run apt-get install -y libegl1 libglvnd0 libglx0 libopengl0
# if command -v vulkaninfo >/dev/null 2>&1; then
#   imagewam_run vulkaninfo --summary
# fi

imagewam_require_env ROBOTWIN_DIR
imagewam_print_config ROBOTWIN_DIR ROBOTWIN_VENV TRANSFORMERS_VERSION

# 1. Lock-exact base. UV_PROJECT_ENVIRONMENT is the only way to get uv's cu128
#    [tool.uv.sources] honoured -- `uv pip install torch` ignores it and resolves the cu13
#    wheel instead. Scoped to this one command; activation owns VIRTUAL_ENV below.
#
#    `uv sync` is destructive: it removes anything not in uv.lock, so it runs first and
#    re-running it alone would wipe steps 3-5. Re-running this whole script is safe.
export UV_PROJECT_ENVIRONMENT="${ROBOTWIN_VENV}"
imagewam_run uv sync --python 3.11 --extra shared
unset UV_PROJECT_ENVIRONMENT

# 2. Activation is required here, not just PYTHON_BIN: the vendored installers below call
#    bare `uv pip install`, which targets VIRTUAL_ENV and otherwise walks up to the main .venv.
# shellcheck disable=SC1091
source "${ROBOTWIN_VENV}/bin/activate"
PYTHON_BIN="${ROBOTWIN_VENV}/bin/python"
export PYTHON_BIN

# 3. RoboTwin packages: requirements, sapien/mplib source patches, warp-lang, curobo.
#    First, wheel: curobo is built with --no-build-isolation, so it uses this env's own
#    tooling. uv sync does not provide wheel (it lives in the `dim` extra, not `shared`) and
#    the vendored installer only adds setuptools, so setuptools has no bdist_wheel command
#    and the build dies with "invalid command 'bdist_wheel'". This must follow uv sync,
#    which would otherwise remove it again.
imagewam_run uv pip install wheel

#    The packages themselves. Skipped on re-runs because the curobo build dominates;
#    FORCE_ROBOTWIN_INSTALL=1 redoes it. Note curobo's *distribution* name differs from its
#    import name -- it installs as `nvidia-curobo`, so `uv pip show curobo` never matches
#    and would silently turn this guard into an unconditional rebuild.
if [ -n "${FORCE_ROBOTWIN_INSTALL:-}" ] || ! uv pip show sapien >/dev/null 2>&1 || ! uv pip show nvidia-curobo >/dev/null 2>&1; then
  (cd "${ROBOTWIN_DIR}" && imagewam_run bash script/_install_uv.sh) || true
else
  echo "[robotwin] sapien/curobo already present; skipping _install_uv.sh"
fi

# 4. Assets. Skipped when already unpacked: unzip would prompt to overwrite, and the script
#    then rm -rf's the archive it could not read. stdin is closed so the one input() in
#    update_embodiment_config_path.py fails fast rather than hanging, if we ever reach it.
if [ ! -d "${ROBOTWIN_DIR}/assets/embodiments" ]; then
  (cd "${ROBOTWIN_DIR}" && imagewam_run bash script/_download_assets.sh </dev/null) || true
else
  echo "[robotwin] assets already present; skipping _download_assets.sh"
fi

# 5. The one deliberate divergence from uv.lock, applied last so nothing re-resolves it.
#    transformers 4.56.1 also lifts huggingface-hub to >=0.34 and tokenizers to >=0.22.
imagewam_run uv pip install "transformers==${TRANSFORMERS_VERSION}"

# 6. The vendored installers have no `set -e` and end in `echo`, so they exit 0 even when
#    they fail -- this is the only real check that steps 3 and 4 actually worked.
ROBOTWIN_DIR="${ROBOTWIN_DIR}" ROBOTWIN_VENV="${ROBOTWIN_VENV}" TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION}" imagewam_run imagewam_python - <<'CHECKPY'
import importlib.util
import os
import sys
from pathlib import Path

problems = []

import torch

# Parity with the main .venv, which is the cu128 build.
if torch.version.cuda != "12.8":
    problems.append(
        f"torch is not the cu128 build: {torch.__version__} (cuda {torch.version.cuda})"
    )
if not torch.cuda.is_available():
    problems.append("torch.cuda.is_available() is False")

import transformers

want = os.environ["TRANSFORMERS_VERSION"]
if transformers.__version__ != want:
    problems.append(f"transformers {transformers.__version__} != {want}")

# find_spec rather than import: curobo initialises Warp on import, which is slow.
for mod in ("torchcodec", "sapien", "mplib", "curobo", "warp"):
    if importlib.util.find_spec(mod) is None:
        problems.append(f"{mod} is not installed")

embodiments = Path(os.environ["ROBOTWIN_DIR"]) / "assets" / "embodiments"
# The *_tmp.yml files are templates that keep ${ASSETS_PATH} by design; only the generated
# siblings have it substituted. Matching on rglob("*.yml") alone would also match the
# templates and pass even if nothing was rewritten.
generated = [p for p in embodiments.rglob("*.yml") if not p.name.endswith("_tmp.yml")]
if not embodiments.is_dir():
    problems.append(f"{embodiments} missing -- the asset download did not complete")
elif not generated:
    problems.append("no generated embodiment configs (only *_tmp.yml templates found)")
elif any("${ASSETS_PATH}" in p.read_text(errors="ignore") for p in generated):
    problems.append("generated embodiment configs still contain ${ASSETS_PATH}")

if problems:
    print("[robotwin] environment check failed:", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    sys.exit(1)
print(f"[robotwin] {os.environ['ROBOTWIN_VENV']} ready")
CHECKPY
