#!/usr/bin/env python3
import rospy
import random
import torch
import time
import cv2
import numpy as np
from datetime import datetime
from collections import deque

import hydra
from omegaconf import OmegaConf

# Import your PiZeroInference and any needed utilities
# Make sure pizero.py is on your PYTHONPATH or in the same directory
# e.g. from my_robot_inference.pizero import PiZeroInference
from src.model.vla.pizero import PiZeroInference
from src.agent.env_adapter.simpler import EDRSimplerAdapter

from rosif import RosIf

def loginfo(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}][INFO] {msg}")

def logwarn(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}][WARN] {msg}")    

def logerr(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}][ERROR] {msg}")  

#===============================
# Utility for loading checkpoint
#===============================
def load_checkpoint(model, checkpoint_path):
    """
    Similar to the snippet in try_checkpoint_in_simpler.py
    """
    # 'weights_only=True' in your example, but if your .pt might have a different structure, adapt as needed.
    ckpt_data = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    if "model" in ckpt_data:
        # Remove any prefix if saved model was compiled
        ckpt_data["model"] = {k.replace("_orig_mod.", ""): v for k, v in ckpt_data["model"].items()}
        model.load_state_dict(ckpt_data["model"], strict=True)
    else:
        # else assume the entire state_dict is in ckpt_data
        model.load_state_dict(ckpt_data, strict=True)
    loginfo(f"PiZeroInference model loaded from {checkpoint_path}")


class GoogleRobotOpenPiZeroInferenceNode:
    def __init__(self, node_name):
        self.rosif = RosIf(node_name)
        self.rosif.text_instruction = "Pick a coke-can"

        # --- ROS Params ---
        self.loop_rate_hz = rospy.get_param("~loop_rate_hz", 3)
        self.checkpoint_path = rospy.get_param("~checkpoint_path", "/root/workspace/dataset/vla_log/2025-01-20_00-22_42_fractal_beta/checkpoint/step147900.pt")
        self.config_path = rospy.get_param("~config_path", "/root/workspace/open-pi-zero/config/eval/fractal_apple.yaml")
        self.gpu_id = rospy.get_param("~gpu_id", 0)
        self.use_bf16 = rospy.get_param("~use_bf16", True)
        self.use_torch_compile = rospy.get_param("~use_torch_compile", True)
        flow_sampling = rospy.get_param("~flow_sampling", "beta") # "beta" or "uniform"

        loginfo(f"loop_rate_hz: {self.loop_rate_hz}")

        # seeding
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        # Prepare device + dtype
        self.device = f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.use_bf16 else torch.float32

        # Load the PiZero config from a YAML or .py-based Hydra config
        self.cfg = OmegaConf.load(self.config_path)  # e.g. "config/eval/bridge.yaml"
        loginfo(f"[PiZero] Loaded config from {self.config_path}")
        self.cfg.env.adapter.dataset_statistics_path = f"/workspaces/open-pi-zero/{self.cfg.env.adapter.dataset_statistics_path}"

        # determine flow matching schedule
        self.cfg.flow_sampling = flow_sampling

        # Initialize the PiZero model
        self.model = PiZeroInference(self.cfg, use_ddp=False)
        load_checkpoint(self.model, self.checkpoint_path)
        # Freeze all weights (inference only)
        self.model.freeze_all_weights()
        self.model.to(self.dtype)
        self.model.to(self.device)
        if self.use_torch_compile:
            self.model = torch.compile(self.model, mode="default")
        self.model.eval()
        loginfo(f"[PiZero] Model moved to {self.device} with dtype={self.dtype}")

        self.adapter:EDRSimplerAdapter = hydra.utils.instantiate(self.cfg.env.adapter)
        self.adapter.reset()

        # For image resizing
        self.model_input_width = self.cfg.env.adapter.image_size[0]
        self.model_input_height = self.cfg.env.adapter.image_size[1]
        loginfo(f"Image Size w:{self.model_input_width}, h:{self.model_input_height}")

        # We track inference times in a sliding window to log stats
        self.inference_time_buffer = deque(maxlen=20)
        self.timer_count = 0

        # Start a periodic timer to run inference + publish commands
        rospy.Timer(rospy.Duration(1.0 / self.loop_rate_hz), self._control_loop)

    def _convert_image(self):
        """
        We'll keep it in HWC format so we can easily resize or process with OpenCV.
        """
        try:
            msg = self.rosif.latest_image
            height, width = msg.height, msg.width
            channels = 3  # RGB8
            if msg.encoding == 'rgb8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, channels))
            else:
                raise Exception(f"image format {msg.encoding} is not supported yet")
            image_out = img  # shape (H, W, 3) RGB8
        except Exception as e:
            logwarn(f"Failed to convert camera image: {e}")
            image_out = None
        return image_out

    def _prepare_model_inputs(self, eef_pos, eef_quat_xyzw, gripper_closedness):
        """
        Manually build PiZero model inputs without calling simpler.py's 'preprocess()'.
        We'll do:
          - Image => resized to model's input size
          - Dummy text tokens & attention masks
          - Normalized 8D proprio if your model training expects that
        """
        latest_image = self._convert_image()
        # latest_image is shape (H,W,3) RGB
        # Resize to your model's input resolution
        resized_image = cv2.resize(
            latest_image,
            (self.cfg.env.adapter.image_size[0], self.cfg.env.adapter.image_size[1]),
            interpolation=cv2.INTER_AREA
        )
        # Convert to torch (B=1,3,H,W)
        images = torch.as_tensor(resized_image, dtype=torch.uint8).permute(2, 0, 1)[
            None
        ] # [1, 3, H, W]
        instruction = self.rosif.text_instruction
        model_inputs = self.adapter.processor(text=[instruction], images=images)

        # 3) Build normalized proprio (8D)
        #    If your model was trained with [-1..1] bounding, do the same here.
        raw_proprio = np.concatenate([eef_pos, eef_quat_xyzw, [gripper_closedness]], axis=0)  # shape (3+4+1=8,)

        # normalize proprios - gripper opening is normalized
        if self.adapter.proprio_normalization_type == "bound":
            proprio = self.adapter.normalize_bound(
                raw_proprio,
                np.array(self.adapter.dataset_statistics["proprio"]["p01"]),
                np.array(self.adapter.dataset_statistics["proprio"]["p99"]),
                clip_min=-1,
                clip_max=1,
            )
        elif self.adapter.proprio_normalization_type == "gaussian":
            proprio = self.adapter.normalize_gaussian(
                raw_proprio,
                np.array(self.dataset_statistics["proprio"]["mean"]),
                np.array(self.dataset_statistics["proprio"]["std"]),
            )

        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
            self.model.build_causal_mask_and_position_ids(
                model_inputs["attention_mask"], dtype=self.dtype
            )
        )

        image_text_proprio_mask, action_mask = self.model.split_full_mask_into_submasks(
            causal_mask
        )

        inputs = {
            "input_ids": model_inputs["input_ids"],
            "pixel_values": model_inputs["pixel_values"].to(self.dtype),
            "image_text_proprio_mask": image_text_proprio_mask,
            "action_mask": action_mask,
            "vlm_position_ids": vlm_position_ids,
            "proprio_position_ids": proprio_position_ids,
            "action_position_ids": action_position_ids,
            "proprios": torch.as_tensor(proprio, dtype=torch.float32)[
                None, None
            ].to(self.dtype), # [B, T, dim]
        }
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def _control_loop(self, _event):
        """
        Called periodically at ~loop_rate_hz to:
          - Read EEF pose from TF + finger angles => build 8D 'proprio'
          - Prepare PiZero model inputs (img, text, mask, proprio, etc.)
          - Run the PiZero inference => 7D EEF delta + gripper delta
          - Convert EEF delta to new EEF pose
          - Solve IK (or use a PD EEF controller) to get joint angles
          - Send the resulting joint command to the trajectory controller
        """

        if not self.rosif.is_observation_available():
            return

        self.timer_count += 1
        start_t = time.time()

        # Gather EEF pose from TF, and gripper state
        (eef_pos, eef_quat_xyzw) = self.rosif.lookup_transform("link_base", "link_gripper")
        gripper_closedness = self.rosif.get_gripper_proprio()

        # Prepare the model inputs
        model_inputs = self._prepare_model_inputs(eef_pos, eef_quat_xyzw, gripper_closedness)

        # Forward pass, Run PiZero inference
        with torch.inference_mode():
            predicted_actions = self.model(**model_inputs)  # -> shape [B, horizon_steps, 7]
        infer_dt = time.time() - start_t

        # Log inference time stats
        self.inference_time_buffer.append(infer_dt)
        if self.timer_count % 10 == 0:
            t_arr = np.array(self.inference_time_buffer, dtype=np.float32)
            loginfo(
                f"[PiZero Inference] avg={t_arr.mean():.3f}s, min={t_arr.min():.3f}s, max={t_arr.max():.3f}s, n={len(t_arr)}"
            )
        predicted_actions = predicted_actions[0].to(torch.float32).cpu().numpy() # shape (T,7), convert BF16 to FP32

        # Denormalize action, gripper action is not normalized in training dataset
        if self.adapter.action_normalization_type == "bound":
            raw_actions_except_gripper = self.adapter.denormalize_bound(
                predicted_actions[:, :-1],
                np.array(self.adapter.dataset_statistics["action"]["p01"])[:-1],
                np.array(self.adapter.dataset_statistics["action"]["p99"])[:-1],
                clip_min=-1,
                clip_max=1,
            )
        elif self.adapter.action_normalization_type == "gaussian":
            raw_actions_except_gripper = self.adapter.denormalize_gaussian(
                predicted_actions[:, :-1],
                np.array(self.adapter.dataset_statistics["action"]["mean"])[:-1],
                np.array(self.adapter.dataset_statistics["action"]["std"])[:-1],
            )
        raw_actions = np.concatenate(
            [
                raw_actions_except_gripper,
                predicted_actions[:, -1:],
            ],
            axis=1,
        )
        count = 1
        for eef_delta in raw_actions[: self.cfg.act_steps]: # in fractal, usually act_steps = 2
            # Convert from [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, Δgrip] into new EEF pose + new gripper value
            # For simplicity, let's parse them
            dx, dy, dz, roll, pitch, yaw, gripper_openness = eef_delta
            self.rosif.control_gripper_by_action(gripper_openness) # 1 for close, 0 for open
            print(f"[{count}] {gripper_openness}")
            count += 1
            for l in range(2):
                self.rosif.apply_action_with_servo(dx, dy, dz, roll, pitch, yaw, 10.0)
                self.rosif.sleep_spin(0.1) 

        self.rosif.clear_observation()

    def spin(self):
        rospy.spin()

def main():
    node = GoogleRobotOpenPiZeroInferenceNode("open_pizero_inference_node")

    # move to initial posture
    node.rosif.switch_controllers("trajectory")
    node.rosif.move_initial_pose()
    time.sleep(3)
    node.rosif.switch_controllers("position")

    node.spin()

if __name__ == "__main__":
    main()
