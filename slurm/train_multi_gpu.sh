#!/bin/bash

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
NUM_GPU="$(nvidia-smi --list-gpus | wc -l)"
echo "NUM_GPU=$NUM_GPU"

# find a free port for torchrun's rendezvous
export MASTER_ADDR=127.0.0.1
function find_free_port() {
    python -c "import socket; s=socket.socket(socket.AF_INET,socket.SOCK_STREAM); s.bind(('',0)); \
               port=s.getsockname()[1]; s.close(); print(port)"
}
export MASTER_PORT=$(find_free_port)

HYDRA_FULL_ERROR=1 uv run \
  torchrun \
  --nnodes=1 \
  --nproc_per_node=$NUM_GPU \
  --rdzv_id=$RANDOM \
  --rdzv_backend=c10d \
  --standalone \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  scripts/run.py \
    --config-name=fractal \
    wandb=null \
    device=cuda:0 \
    global_batch_size=256 \
    per_device_batch_size=32 \
    action_lr=0.00005 \
    vlm_lr=0.00005 \
    flow_sampling=beta \
    data.train.shuffle_buffer_size=40000 \
    data.train.num_parallel_calls=64 \
    eval_freq=2000 \
    eval_size=256 \
    save_model_freq=10000 \
    save_model_start=0 \
    use_torch_compile=True \
    use_bf16=True \
    use_amp=True \
    use_swa=False \
    lora=False \
    quantize=False

