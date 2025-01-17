#!/bin/bash
# This script is used to install the required packages for open-pi-zero project
# when you build Dev container for the first time.
set -e
# set root directory of open-pi-zero project
SCRIPT_DIR=$(dirname "$(realpath $0)")
PRJ_DIR=$(realpath ${SCRIPT_DIR}/../)

# Clone SimplerEnv in parallel to open-pi-zero
git clone https://github.com/allenzren/SimplerEnv --recurse-submodules $(realpath ${PRJ_DIR}/../SimplerEnv)

# And go back to open-pi-zero project and install the required packages
cd ${PRJ_DIR}
uv sync
uv pip install -e . ../SimplerEnv --no-config
uv pip install -e . ../SimplerEnv/ManiSkill2_real2sim --no-config

# Adding the following line to your ~/.bashrc to activate the virtual environment automatically
#echo "source ${PRJ_DIR}/.venv/bin/activate" >> ~/.bashrc
