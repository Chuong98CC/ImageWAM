#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

LIBERO_DIR="${LIBERO_DIR:-${REPO_ROOT}/third_party/LIBERO}"
LIBERO_REPO="${LIBERO_REPO:-https://github.com/Lifelong-Robot-Learning/LIBERO.git}"
LIBERO_CONFIG_DIR="${LIBERO_CONFIG_DIR:-${HOME}/.libero}"
# LIBERO and LIBERO-plus both install a package named "libero", so they cannot share a venv
# without reinstalling on every switch. Each gets its own: this one lives in LIBERO_VENV
# (.venv), the plus one in LIBERO_PLUS_VENV (.venv_libero_plus). LIBERO_CONFIG_DIR keeps their
# asset/bddl/init-state paths apart the same way, since those are read from a single config file.
LIBERO_VENV="${LIBERO_VENV:-${REPO_ROOT}/.venv}"
if [ ! -x "${LIBERO_VENV}/bin/python" ]; then
  echo "LIBERO venv not found: ${LIBERO_VENV}" >&2
  echo "Create it with: uv sync --python 3.11 --extra shared" >&2
  exit 2
fi

# System libs need root. libgl1-mesa-glx does not exist on Ubuntu 24.04; libgl1
# is the package that provides libGL.so.1 there (and on older releases too).
APT_PREFIX=()
if [ "$(id -u)" -ne 0 ]; then
  APT_PREFIX=(sudo)
fi
# imagewam_run "${APT_PREFIX[@]}" apt-get update
# imagewam_run "${APT_PREFIX[@]}" apt-get install -y libosmesa6-dev libgl1 libglfw3
imagewam_run uv pip install --python "${LIBERO_VENV}/bin/python" mujoco==3.3.2 robosuite==1.4.0 bddl==1.0.1 gym==0.25.2 easydict thop future cloudpickle opencv-python-headless

if [ ! -d "${LIBERO_DIR}" ]; then
  mkdir -p "$(dirname "${LIBERO_DIR}")"
  imagewam_run git clone "${LIBERO_REPO}" "${LIBERO_DIR}"
fi

# setuptools' find_packages() needs an __init__.py at EVERY level. Upstream tracks
# only the inner one, so without the outer libero/__init__.py find_packages() returns
# nothing, the editable install exposes no packages, and eval dies with
# "No module named 'libero'".
touch "${LIBERO_DIR}/libero/__init__.py"
touch "${LIBERO_DIR}/libero/libero/__init__.py"
LIBERO_BENCHMARK_INIT="${LIBERO_DIR}/libero/libero/benchmark/__init__.py" imagewam_run "${LIBERO_VENV}/bin/python" - <<'PATCHPY'
import os
from pathlib import Path
path = Path(os.environ['LIBERO_BENCHMARK_INIT'])
text = path.read_text()
old = 'init_states = torch.load(init_states_path)'
new = 'init_states = torch.load(init_states_path, weights_only=False)'
if old in text:
    path.write_text(text.replace(old, new))
elif new not in text:
    raise RuntimeError(f'Could not patch torch.load in {path}')
PATCHPY

(cd "${LIBERO_DIR}" && imagewam_run uv pip install --python "${LIBERO_VENV}/bin/python" -e . --force-reinstall)
mkdir -p "${LIBERO_CONFIG_DIR}/config_backups"
cp "${LIBERO_CONFIG_DIR}/config.yaml" "${LIBERO_CONFIG_DIR}/config_backups/config.$(date +%Y%m%d_%H%M%S).yaml" 2>/dev/null || true
rm -f "${LIBERO_CONFIG_DIR}/config.yaml"

# LIBERO's package __init__ calls input() when config.yaml is missing, which aborts
# headless eval workers with EOFError. Import it against a scratch config dir to skip
# that prompt, then write the real config using LIBERO's own default-path logic.
LIBERO_CONFIG_DIR="${LIBERO_CONFIG_DIR}" LIBERO_PKG_DIR="${LIBERO_DIR}/libero/libero" imagewam_run "${LIBERO_VENV}/bin/python" - <<'PATCHPY'
import os
import tempfile
from pathlib import Path

import yaml

scratch = Path(tempfile.mkdtemp())
(scratch / "config.yaml").write_text("{}\n")
os.environ["LIBERO_CONFIG_PATH"] = str(scratch)

import libero.libero as libero

config_file = Path(os.environ["LIBERO_CONFIG_DIR"]) / "config.yaml"
config_file.parent.mkdir(parents=True, exist_ok=True)
with config_file.open("w") as f:
    yaml.dump(libero.get_default_path_dict(os.environ["LIBERO_PKG_DIR"]), f)
print(f"[libero] wrote {config_file}")
PATCHPY
