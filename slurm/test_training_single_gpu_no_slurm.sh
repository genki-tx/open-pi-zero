#!/bin/bash

# no wandb logging
# logging GPU memory usage
# smaller global batch size
# small resource for dataloading
# try saving model

# first batch will take a while with torch.compile as model being compiled
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 uv run \
    scripts/run.py \
    --config-name=fractal \
    device=cuda:0 \
    debug=True \
    wandb=null \
    log_dir=results/test/ \
    global_batch_size=4 \
    per_device_batch_size=4 \
    flow_sampling=beta \
    data.train.shuffle_buffer_size=1000 \
    data.train.num_parallel_calls=4 \
    eval_freq=128 \
    eval_size=16 \
    save_model_freq=100 \
    save_model_start=0 \
    lora=True \
    quantize=False \
    use_torch_compile=True \
    use_bf16=True \
    use_amp=True \
    use_ema=False \
    ema_decay=0.99 \
    ema_device=cuda \
    use_swa=False \
    swa_start=0 \
    swa_freq=2 \
    swa_device=cpu \
    action_lr_scheduler.warmup_steps=0 \
    vlm_lr_scheduler.warmup_steps=0
# 'resume_checkpoint_path="...fractal_train[:95%]_tp4_beta...ckpt....pt"'
