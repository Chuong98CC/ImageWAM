#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

LIBERO_PLUS_DIR="${LIBERO_PLUS_DIR:-${REPO_ROOT}/third_party/LIBERO-plus}"
LIBERO_PLUS_REPO="${LIBERO_PLUS_REPO:-https://github.com/sylvestf/LIBERO-plus.git}"
LIBERO_CONFIG_DIR="${LIBERO_CONFIG_DIR:-${HOME}/.libero}"
LIBERO_PLUS_PKG_DIR="${LIBERO_PLUS_DIR}/libero/libero"
LIBERO_PLUS_ASSETS_DIR="${LIBERO_PLUS_PKG_DIR}/assets"
LIBERO_PLUS_NESTED_ASSETS_DIR="${LIBERO_PLUS_PKG_DIR}/inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets"

# System libs need root. libgl1-mesa-glx does not exist on Ubuntu 24.04; libgl1
# is the package that provides libGL.so.1 there (and on older releases too).
APT_PREFIX=()
if [ "$(id -u)" -ne 0 ]; then
  APT_PREFIX=(sudo)
fi
imagewam_run "${APT_PREFIX[@]}" apt-get install -y libosmesa6-dev libgl1 libglfw3 unzip libexpat1 libfontconfig1-dev libpython3-stdlib libmagickwand-dev
imagewam_run uv pip install mujoco==3.3.2 robosuite==1.4.0 bddl==1.0.1 gym==0.25.2 easydict thop future cloudpickle opencv-python-headless scikit-image wand

if [ ! -d "${LIBERO_PLUS_DIR}" ]; then
  mkdir -p "$(dirname "${LIBERO_PLUS_DIR}")"
  imagewam_run git clone "${LIBERO_PLUS_REPO}" "${LIBERO_PLUS_DIR}"
fi

if [ ! -d "${LIBERO_PLUS_ASSETS_DIR}" ]; then
  imagewam_run huggingface-cli download Sylvest/LIBERO-plus assets.zip --repo-type dataset --local-dir "${LIBERO_PLUS_PKG_DIR}"
  imagewam_run unzip -q -o "${LIBERO_PLUS_PKG_DIR}/assets.zip" -d "${LIBERO_PLUS_PKG_DIR}"
  if [ -d "${LIBERO_PLUS_NESTED_ASSETS_DIR}" ]; then
    imagewam_run mv "${LIBERO_PLUS_NESTED_ASSETS_DIR}" "${LIBERO_PLUS_ASSETS_DIR}"
  fi
fi

touch "${LIBERO_PLUS_DIR}/libero/__init__.py"
LIBERO_PLUS_BENCHMARK_INIT="${LIBERO_PLUS_PKG_DIR}/benchmark/__init__.py" imagewam_run imagewam_python - <<'PATCHPY'
import os
from pathlib import Path
path = Path(os.environ['LIBERO_PLUS_BENCHMARK_INIT'])
text = path.read_text()
old = 'init_states = torch.load(init_states_path)'
new = 'init_states = torch.load(init_states_path, weights_only=False)'
if old in text:
    path.write_text(text.replace(old, new))
elif new not in text:
    raise RuntimeError(f'Could not patch torch.load in {path}')
PATCHPY

LIBERO_PLUS_ENV_WRAPPER="${LIBERO_PLUS_PKG_DIR}/envs/env_wrapper.py" imagewam_run imagewam_python - <<'PATCHPY'
import os
from pathlib import Path
path = Path(os.environ['LIBERO_PLUS_ENV_WRAPPER'])
text = path.read_text()
old = '    ):\n        if "_view_" in bddl_file_name and "_initstate_" in bddl_file_name:'
new = '    ):\n        bddl_file_name = str(bddl_file_name)\n        if "_view_" in bddl_file_name and "_initstate_" in bddl_file_name:'
if old in text:
    path.write_text(text.replace(old, new))
elif new not in text:
    raise RuntimeError(f'Could not patch bddl_file_name Path handling in {path}')
PATCHPY

(cd "${LIBERO_PLUS_DIR}" && imagewam_run uv pip install -e .)
mkdir -p "${LIBERO_CONFIG_DIR}/config_backups"
cp "${LIBERO_CONFIG_DIR}/config.yaml" "${LIBERO_CONFIG_DIR}/config_backups/config.$(date +%Y%m%d_%H%M%S).yaml" 2>/dev/null || true
rm -f "${LIBERO_CONFIG_DIR}/config.yaml"

# LIBERO's package __init__ calls input() when config.yaml is missing, which aborts
# headless eval workers with EOFError. Import it against a scratch config dir to skip
# that prompt, then write the real config using LIBERO's own default-path logic.
# LIBERO_PKG_DIR is passed explicitly because LIBERO-plus installs a package also
# named "libero", so the import target is ambiguous when both are installed.
LIBERO_CONFIG_DIR="${LIBERO_CONFIG_DIR}" LIBERO_PKG_DIR="${LIBERO_PLUS_PKG_DIR}" imagewam_run imagewam_python - <<'PATCHPY'
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
