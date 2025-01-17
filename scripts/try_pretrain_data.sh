#!/bin/bash
set -e
# modify the path to download the pretrain model to match the path in your environment
DATASET_DIR="${HOME}/dataset"

mkdir -p ${DATASET_DIR}/vla_log
wget -P ~/dataset/vla_log https://huggingface.co/allenzren/open-pi-zero/resolve/main/bridge_uniform_step19296_2024-12-26_22-31_42.pt

