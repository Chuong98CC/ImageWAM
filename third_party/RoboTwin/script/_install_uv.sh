#!/usr/bin/env bash
# Run this from the ImageWAM environment used by the wrapper setup script.
echo "Installing the necessary packages ..."
uv pip install -r script/requirements_mod.txt
uv pip install 'setuptools<81'

apt install libvulkan1 mesa-vulkan-drivers vulkan-tools

echo "Installing pytorch3d ..."
# Build 
unset LC_ALL
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
# ./cuda_11.8.0_520.61.05_linux.run --silent --toolkit 

# PYTORCH3D_BUILD_JOBS=${PYTORCH3D_BUILD_JOBS:-192}
# According to RoboTwin, pytorch3d is not required if depth & 3D utilities are not needed.
# echo "Building pytorch3d with ${PYTORCH3D_BUILD_JOBS} parallel jobs ..."
# MAX_JOBS="${PYTORCH3D_BUILD_JOBS}" CMAKE_BUILD_PARALLEL_LEVEL="${PYTORCH3D_BUILD_JOBS}" \
#     uv pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation

echo "Adjusting code in sapien/wrapper/urdf_loader.py ..."
# location of sapien, like "~/.conda/envs/RoboTwin/lib/python3.10/site-packages/sapien"
SAPIEN_LOCATION=$(uv pip show sapien | grep 'Location' | awk '{print $2}')/sapien
# Adjust some code in wrapper/urdf_loader.py
URDF_LOADER=$SAPIEN_LOCATION/wrapper/urdf_loader.py
# ----------- before -----------
# 667         with open(urdf_file, "r") as f:
# 668             urdf_string = f.read()
# 669 
# 670         if srdf_file is None:
# 671             srdf_file = urdf_file[:-4] + "srdf"
# 672         if os.path.isfile(srdf_file):
# 673             with open(srdf_file, "r") as f:
# 674                 self.ignore_pairs = self.parse_srdf(f.read())
# ----------- after  -----------
# 667         with open(urdf_file, "r", encoding="utf-8") as f:
# 668             urdf_string = f.read()
# 669 
# 670         if srdf_file is None:
# 671             srdf_file = urdf_file[:-4] + ".srdf"
# 672         if os.path.isfile(srdf_file):
# 673             with open(srdf_file, "r", encoding="utf-8") as f:
# 674                 self.ignore_pairs = self.parse_srdf(f.read())
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' $URDF_LOADER


echo "Adjusting code in mplib/planner.py ..."
# location of mplib, like "~/.conda/envs/RoboTwin/lib/python3.10/site-packages/mplib"
MPLIB_LOCATION=$(uv pip show mplib | grep 'Location' | awk '{print $2}')/mplib

# Adjust some code in planner.py
# ----------- before -----------
# 807             if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:
# 808                 return {"status": "screw plan failed"}
# ----------- after  ----------- 
# 807             if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:
# 808                 return {"status": "screw plan failed"}
PLANNER=$MPLIB_LOCATION/planner.py
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' $PLANNER

uv pip install warp-lang==1.12.0

echo "Installing Curobo ..."
cd envs
git clone https://github.com/NVlabs/curobo.git
cd curobo
git fetch --tags
git checkout v0.7.8
# Capped from the upstream 192: these are literal assignments, so exporting cannot override
# them, and 192 parallel compiles oversubscribes smaller build hosts into an OOM.
MAX_JOBS=16 CMAKE_BUILD_PARALLEL_LEVEL=16 uv pip install -e . --no-build-isolation
cd ../..

echo "Installation basic environment complete!"
echo -e "You need to:"
echo -e "    1. \033[34m\033[1m(Important!)\033[0m Download assets from huggingface."
echo -e "    2. Install requirements for running baselines. (Optional)"
echo "See INSTALLATION.md for more instructions."
