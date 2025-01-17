#!/bin/bash
set -e
# modify the path to download the pretrain model to match the path in your environment
DATASET_DIR="${HOME}/dataset"

# setup the directory for the dataset
HF_DATA_DIR="${DATASET_DIR}/hf_models"
VLA_DATA_DIR="${DATASET_DIR}/vla_data"
VLA_LOG_DIR="${DATASET_DIR}/vla_log"
mkdir -p ${HF_DATA_DIR}
mkdir -p ${VLA_DATA_DIR}
mkdir -p ${VLA_LOG_DIR}

# set download_datasets=true if you want to download, set false if you skip it
download_datasets=true
if [ "$download_datasets" = true ]; then
    # download paligemma from hugging face, setup your account with huggingface-cli first, https://huggingface.co/docs/huggingface_hub/en/guides/cli
    huggingface-cli download google/paligemma-3b-pt-224 --local-dir ${HF_DATA_DIR}/paligemma-3b-pt-224

    # download fractal, setup gsutil command, https://cloud.google.com/storage/docs/gsutil_install
    gsutil -m cp -n -r gs://gresearch/robotics/fractal20220817_data/0.1.0 ${VLA_DATA_DIR}/

    # download bridge dataset
    wget -r -np -nH --cut-dirs=3 -R "index.html*" -nc -P ${VLA_DATA_DIR}/bridge_dataset https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/

fi # download_datasets

# if you prefer to download pretrain mode, set download_pretrain_model=true 
download_pretrain_model=true
if [ "$download_pretrain_model" = true ]; then
    wget -nc -P ${VLA_LOG_DIR} https://huggingface.co/allenzren/open-pi-zero/resolve/main/bridge_uniform_step19296_2024-12-26_22-31_42.pt
    wget -nc -P ${VLA_LOG_DIR} https://huggingface.co/allenzren/open-pi-zero/resolve/main/bridge_beta_step19296_2024-12-26_22-30_42.pt
    wget -nc -P ${VLA_LOG_DIR} https://huggingface.co/allenzren/open-pi-zero/resolve/main/fractal_uniform_step29576_2024-12-31_22-26_42.pt
    wget -nc -P ${VLA_LOG_DIR} https://huggingface.co/allenzren/open-pi-zero/resolve/main/fractal_beta_step29576_2024-12-29_13-10_42.pt
fi

