import os
import random
import time

import hydra
import imageio
import numpy as np
import simpler_env
import torch
from omegaconf import OmegaConf

from src.model.vla.pizero import PiZeroInference
from src.utils.monitor import log_allocated_gpu_memory, log_execution_time


@log_execution_time()
def load_checkpoint(model, path):
    """Load checkpoint (weights_only=True if the .pt was saved that way)."""
    data = torch.load(path, weights_only=True, map_location="cpu")
    # Remove "_orig_mod." prefix if saved model was compiled
    data["model"] = {k.replace("_orig_mod.", ""): v for k, v in data["model"].items()}
    model.load_state_dict(data["model"], strict=True)
    print(f"Loaded model from {path}")


def run_one_episode(args, model, cfg, seed, device, dtype):
    """
    Run exactly ONE episode with a given seed.
    Mirrors your original main(args) logic but for a single seed.
    """
    # Seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create environment (same as original)
    env = simpler_env.make(args.task)

    # Create/instantiate the environment adapter
    env_adapter = hydra.utils.instantiate(cfg.env.adapter)
    env_adapter.reset()

    # Reset environment
    episode_id = random.randint(0, 20)
    env_reset_options = {"obj_init_options": {"episode_id": episode_id}}
    obs, reset_info = env.reset(options=env_reset_options)
    instruction = env.get_language_instruction()

    # Optional: video recording
    video_writer = None
    if args.recording:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        video_path = f"try_{args.task}_seed{seed}_episode{episode_id}.mp4"
        video_writer = imageio.get_writer(video_path)

    print(
        f"[Seed={seed}] Reset info: {reset_info} "
        f"Instruction: {instruction} "
        f"Max episode length: {env.spec.max_episode_steps}"
    )
    cnt_step = 0
    inference_times = []
    success = False  # track final success from env

    # Main loop (same as original)
    while True:
        # Preprocess observation
        inputs = env_adapter.preprocess(env, obs, instruction)
        # Build mask/positions
        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
            model.build_causal_mask_and_position_ids(inputs["attention_mask"], dtype=dtype)
        )
        image_text_proprio_mask, action_mask = model.split_full_mask_into_submasks(causal_mask)
        inputs = {
            "input_ids": inputs["input_ids"],
            "pixel_values": inputs["pixel_values"].to(dtype),
            "image_text_proprio_mask": image_text_proprio_mask,
            "action_mask": action_mask,
            "vlm_position_ids": vlm_position_ids,
            "proprio_position_ids": proprio_position_ids,
            "action_position_ids": action_position_ids,
            "proprios": inputs["proprios"].to(dtype),
        }
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Inference
        start_inference_time = time.time()
        with torch.inference_mode():
            actions = model(**inputs)
        # If you want to exclude the first step from inference time average:
        if cnt_step > 0:
            inference_times.append(time.time() - start_inference_time)

        # Post-process -> environment actions
        env_actions = env_adapter.postprocess(actions[0].float().cpu().numpy())

        # Step environment
        for env_action in env_actions[: cfg.act_steps]:
            obs, reward, success, truncated, info = env.step(env_action)
            cnt_step += 1
            if truncated:
                break

        # Save frame
        if video_writer is not None:
            video_writer.append_data(env_adapter.get_video_frame(env, obs))

        # Possibly update instruction
        new_instruction = env.get_language_instruction()
        if new_instruction != instruction:
            instruction = new_instruction

        # Episode end
        if truncated:
            if video_writer is not None:
                video_writer.close()
            break

    # Print summary
    if len(inference_times) > 0:
        avg_inf_time = np.mean(inference_times)
    else:
        avg_inf_time = 0.0

    print("\n\n============ Summary ============")
    print(f"Seed: {seed}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Action chunk steps (predicted): {cfg.horizon_steps}")
    print(f"Action chunk steps (executed): {cfg.act_steps}")
    print(f"Avg inference time (excluding first step): {avg_inf_time:.3f}s")
    print(f"Peak VRAM usage: {torch.cuda.max_memory_reserved(args.gpu_id)/1024**3:.2f} GB")
    print(f"Task: {args.task}")
    print(f"Total environment steps: {cnt_step}")
    print(f"Success: {success}")
    if video_writer is not None:
        print(f"Video saved as {video_path}")
    print("======================================\n\n")

    env.close()


def main_multi_seed(args):
    """
    Load the model once, then run multiple seeds (1..100).
    Identical logic to your original main(), just repeated for different seeds.
    """
    # Load default configs (unchanged)
    if "fractal" in args.checkpoint_path:
        cfg = OmegaConf.load("config/eval/fractal_apple.yaml")
    elif "bridge" in args.checkpoint_path:
        cfg = OmegaConf.load("config/eval/bridge.yaml")
    else:
        raise ValueError("Could not determine fractal/bridge from checkpoint_path")

    if "uniform" in args.checkpoint_path:
        cfg.flow_sampling = "uniform"
    if "beta" in args.checkpoint_path:
        cfg.flow_sampling = "beta"

    # Device / dtype
    device = torch.device(f"cuda:{args.gpu_id}")
    dtype = torch.bfloat16 if args.use_bf16 else torch.float32

    # Create + load model
    model = PiZeroInference(cfg, use_ddp=False)
    load_checkpoint(model, args.checkpoint_path)
    model.freeze_all_weights()
    model.to(dtype).to(device)

    # Torch compile if requested
    if args.use_torch_compile:
        print("Compiling model with torch.compile()...")
        model = torch.compile(model, mode="default")

    model.eval()
    print(f"Using cuda device: {device}, dtype={dtype}")
    log_allocated_gpu_memory(None, "loading model", args.gpu_id)

    # Now, loop over seeds, reusing the same model
    seed_start = 1
    seed_end = args.iteration + seed_start
    for seed in range(seed_start, seed_end):
        run_one_episode(args, model, cfg, seed, device, dtype)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="google_robot_pick_horizontal_coke_can",
        choices=[
            "widowx_carrot_on_plate",
            "widowx_put_eggplant_in_basket",
            "widowx_spoon_on_towel",
            "widowx_stack_cube",
            "google_robot_pick_horizontal_coke_can",
            "google_robot_pick_vertical_coke_can",
            "google_robot_pick_standing_coke_can",
            "google_robot_move_near_v0",
            "google_robot_open_drawer",
            "google_robot_close_drawer",
            "google_robot_place_apple_in_closed_top_drawer",
        ],
    )
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--use_torch_compile", action="store_true")
    parser.add_argument("--recording", action="store_true")
    parser.add_argument("--iteration", type=int, default=10)
    args = parser.parse_args()

    # Quick checks, same as original
    if "google_robot" in args.task:
        assert "fractal" in args.checkpoint_path, (
            "For google_robot tasks, checkpoint_path should contain 'fractal'."
        )
    if "widowx" in args.task:
        assert "bridge" in args.checkpoint_path, (
            "For widowx tasks, checkpoint_path should contain 'bridge'."
        )

    # Run
    main_multi_seed(args)
