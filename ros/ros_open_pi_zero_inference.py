#!/usr/bin/env python3

import rospy
import random
import actionlib
import torch
import time
import cv2
import numpy as np
from datetime import datetime

from collections import deque

from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
import tf
import tf.transformations as tr

import hydra
from omegaconf import OmegaConf

# Import your PiZeroInference and any needed utilities
# Make sure pizero.py is on your PYTHONPATH or in the same directory
# e.g. from my_robot_inference.pizero import PiZeroInference
from src.model.vla.pizero import PiZeroInference

# Workaround for rospy-all deserialization issue in Python 3.10
# ------------------------------------------------------------
# Problem:
# rospy-all expects a "rosmsg" error handler for deserializing complex messages
# (e.g., sensor_msgs/JointState, sensor_msgs/Image). Without this handler,
# deserialization fails, and subscriber callbacks are not triggered.
# This is due to the `codecs.lookup_error("rosmsg")` line in rospy-all's message
# handling code, which expects a custom error handler named "rosmsg" to be registered.
# Fix:
# Register a custom "rosmsg" error handler that re-raises exceptions, allowing
# rospy-all to proceed without errors.
import codecs
# Custom error handler for "rosmsg" deserialization
def rosmsg_error_handler(exception):
    raise exception
# Register the custom error handler
codecs.register_error("rosmsg", rosmsg_error_handler)
# ------------------------------------------------------------

def loginfo(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}][INFO] {msg}")

def logwarn(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}][WARN] {msg}")    

###############################
# Utility for loading checkpoint
###############################
def load_checkpoint(model, checkpoint_path):
    """
    Similar to the snippet in try_checkpoint_in_simpler.py
    """
    # 'weights_only=True' in your example, but if your .pt might have a different structure, adapt as needed.
    ckpt_data = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    # remove "_orig_mod." prefix if saved model was compiled
    ckpt_data["model"] = {k.replace("_orig_mod.", ""): v for k, v in ckpt_data["model"].items()}
    model.load_state_dict(ckpt_data["model"], strict=True)
    loginfo(f"PiZeroInference model loaded from {checkpoint_path}")


class GoogleRobotOpenPiZeroInferenceNode:
    def __init__(self):
        # --- ROS Params ---
        self.loop_rate_hz = rospy.get_param("~loop_rate_hz", 2.0)
        self.checkpoint_path = rospy.get_param("~checkpoint_path", "/root/workspace/dataset/vla_log/2025-01-20_00-22_42_fractal_beta/checkpoint/step147900.pt")
        self.config_path = rospy.get_param("~config_path", "/root/workspace/open-pi-zero/config/eval/fractal_apple.yaml")
        self.gpu_id = rospy.get_param("~gpu_id", 0)
        self.use_bf16 = rospy.get_param("~use_bf16", True)
        self.use_torch_compile = rospy.get_param("~use_torch_compile", True)
        flow_sampling = rospy.get_param("~flow_sampling", "beta") # "beta" or "uniform"

        # For demonstration, these are the relevant joints in "google_arm_controller"
        self.joint_names = [
            "joint_torso", "joint_shoulder", "joint_bicep", "joint_elbow",
            "joint_forearm", "joint_wrist", "joint_gripper",
            "joint_finger_right", "joint_finger_left", "joint_head_pan", "joint_head_tilt"
        ]
        self.num_joints = len(self.joint_names)

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
        # For image resizing
        self.model_input_width = self.cfg.env.adapter.image_size[0]
        self.model_input_height = self.cfg.env.adapter.image_size[1]
        loginfo(f"Image Size w:{self.model_input_width}, h:{self.model_input_height}")

        # We store the latest image and latest joint states
        self.latest_image = None
        self.latest_joints = None

        # We track inference times in a sliding window to log stats
        self.inference_time_buffer = deque(maxlen=20)
        self.timer_count = 0

        # Setup ROS subscriptions
        self.image_sub = rospy.Subscriber("/head_camera/image_raw", Image, self._image_callback, queue_size=1)
        self.joint_state_sub = rospy.Subscriber("/joint_states", JointState, self._joint_state_callback, queue_size=1)
        self.tf_listener = tf.TransformListener()

        # Setup ActionLib client for joint trajectory
        self.client = actionlib.SimpleActionClient("/google_arm_controller/follow_joint_trajectory",
                                                   FollowJointTrajectoryAction)
        loginfo("Waiting for google_arm_controller action server ...")
        self.client.wait_for_server()
        loginfo("Connected to google_arm_controller action server.")

        # Start a periodic timer to run inference + publish commands
        rospy.Timer(rospy.Duration(1.0 / self.loop_rate_hz), self._control_loop)

    def _image_callback(self, msg: Image):
        """
        Convert the camera image to a fixed 224x224 BGR or RGB (depending on your model).
        We'll store as a torch.uint8 (3 x 224 x 224).
        """
        try:
            # if you already have un-precessed value, skip the process to save CPU resources
            #if self.latest_image:
            #    return

            # Extract image dimensions
            width = msg.width
            height = msg.height
            channels = 3  # For BGR8 images

            # Convert image data to a NumPy array
            img_array = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, channels))

            # Resize to model input shape
            resized = cv2.resize(img_array, (self.model_input_width, self.model_input_height), interpolation=cv2.INTER_AREA)

            # Convert to CHW torch.uint8
            resized_chw = np.transpose(resized, (2, 0, 1))  # shape (3, 224, 224)

            # Make a torch tensor
            image_tensor = torch.from_numpy(resized_chw).type(torch.uint8)
            self.latest_image = image_tensor
        except Exception as e:
            logwarn(f"Failed to convert/resize camera image: {e}")
            self.latest_image = None

    def _joint_state_callback(self, msg: JointState):
        """
        Extract relevant joint positions from /joint_states and build a dict
        """
        name_to_pos = {}
        for i, nm in enumerate(msg.name):
            pos = msg.position[i]
            name_to_pos[nm] = pos

        self.latest_joints = name_to_pos

    def _control_loop(self, _event):
        """
        Called periodically at ~loop_rate_hz to:
          1) Read EEF pose from TF + finger angles => build 8D 'proprio'
          2) Prepare PiZero model inputs (img, text, mask, proprio, etc.)
          3) Run the PiZero inference => 7D EEF delta + gripper delta
          4) Convert EEF delta to new EEF pose
          5) Solve IK (or use a PD EEF controller) to get joint angles
          6) Send the resulting joint command to the trajectory controller
        """
        self.timer_count += 1

        # 0) Check we have the latest camera image, finger joint states
        if self.latest_image is None or self.latest_joints is None:
            return

        # 1) Gather EEF pose from TF
        try:
            # If your root frame is "world", and child is "link_gripper_tcp"
            now = rospy.Time(0)
            self.tf_listener.waitForTransform(
                "link_base", "link_gripper_tcp", now, rospy.Duration(0.5)
            )
            (trans, rot) = self.tf_listener.lookupTransform("link_base", "link_gripper_tcp", now)
            # trans => (x,y,z), rot => (qx,qy,qz,qw)
            eef_pos = np.array(trans, dtype=np.float32)
            eef_quat = np.array(rot, dtype=np.float32)
        except Exception as e:
            rospy.logwarn(f"Could not lookup TF for link_gripper_tcp: {e}")
            return

        # 2) Compute 'gripper_opening' from finger joints
        #    Assume 'joint_finger_left' and 'joint_finger_right' each in [0..1.3]
        left_finger = self.latest_joints.get("joint_finger_left", 0.0)
        right_finger = self.latest_joints.get("joint_finger_right", 0.0)
        max_range = 1.3
        # If 0 => open, 1.3 => closed, define 'gripper_opening' in [0..1]
        # Depending on how you used it in training, you might do the inverse
        # E.g. 0=closed, 1=open => adjust to match training.
        gripper_opening = 1.0 - ((left_finger + right_finger) / (2.0 * max_range))

        # 3) Build the 8D 'proprio' for the model
        # Format: [x, y, z, qx, qy, qz, qw, gripper_opening]
        proprio_np = np.concatenate([eef_pos, eef_quat, [gripper_opening]], axis=0)

        # 4) Prepare the model inputs
        model_inputs = self._prepare_model_inputs(
            img_tensor=self.latest_image,   # (3,224,224) uint8
            proprio_np=proprio_np           # shape (8,)
            # If you still want to pass text tokens, etc., adapt _prepare_model_inputs accordingly.
        )

        # 5) Run PiZero inference
        start_t = time.time()
        with torch.inference_mode():
            predicted_actions = self.model(**model_inputs)  # -> shape [B, horizon_steps, 7]
        infer_dt = time.time() - start_t

        # 6) Log inference time stats
        self.inference_time_buffer.append(infer_dt)
        if self.timer_count % 10 == 0:
            t_arr = np.array(self.inference_time_buffer, dtype=np.float32)
            rospy.loginfo(
                f"[PiZero Inference] avg={t_arr.mean():.3f}s, min={t_arr.min():.3f}s, max={t_arr.max():.3f}s, n={len(t_arr)}"
            )

        # Usually B=1, so pick predicted_actions[0] => shape (horizon_steps, 7)
        if predicted_actions.dim() == 3:
            predicted_actions = predicted_actions[0]

        # We'll just take the first step's predicted action
        eef_delta = predicted_actions[0].to(torch.float32).cpu().numpy()  # shape (7,), convert BF16 to FP32

        # 7) Convert from [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, Δgrip] into new EEF pose + new gripper value
        # For simplicity, let's parse them
        dx, dy, dz, droll, dpitch, dyaw, dgrip = eef_delta

        # a) Update position
        new_eef_pos = eef_pos + np.array([dx, dy, dz], dtype=np.float32)

        # b) Update orientation (naive approach in euler)
        # If you have small rotations each step, you can do something like:
        current_euler = np.array(tr.euler_from_quaternion(eef_quat), dtype=np.float32)
        new_euler = current_euler + np.array([droll, dpitch, dyaw], dtype=np.float32)
        new_quat  = tr.quaternion_from_euler(*new_euler).astype(np.float32)

        # c) Update gripper
        new_gripper_opening = np.clip(gripper_opening + dgrip, 0.0, 1.0)

        # 8) Solve IK or feed to an EEF-delta controller
        #    - If you have your own IK, do something like:
        #       new_joint_positions = self.do_ik(new_eef_pos, new_quat)
        #    - Or if you have a PD approach, you might skip the direct IK and just publish a "delta pose" to a custom controller.
        # For demonstration, let's do a dummy "IK" to show how to build a joint trajectory:
        new_joint_positions = self._dummy_ik_solver(new_eef_pos, new_quat, new_gripper_opening)

        # 9) Build a JointTrajectory for google_arm_controller
        traj = JointTrajectory()
        traj.joint_names = self.joint_names  # e.g. 11 joints
        pt = JointTrajectoryPoint()
        pt.positions = new_joint_positions
        pt.time_from_start = rospy.Duration(1.0)
        traj.points.append(pt)

        goal = FollowJointTrajectoryGoal()
        goal.trajectory = traj
        self.client.send_goal(goal)
        # self.client.wait_for_result(rospy.Duration(2.0))

        # 10) Clear the buffers so we wait for next fresh image + fresh finger states
        self.latest_image = None
        self.latest_joints = None

    def _prepare_model_inputs(self, img_tensor: torch.Tensor, proprio_np: np.ndarray):
        """
        Convert the camera image + 8D proprio array into the PiZero model's required format.
        This matches the logic from 'try_checkpoint_in_simpler.py' when using:
           max_image_text_tokens=276 (256 image tokens + up to 20 text tokens).

        :param img_tensor: (3, 224, 224) torch.uint8 from your camera callback.
        :param proprio_np: (8,) np.ndarray = [EEF_x, EEF_y, EEF_z, EEF_quat, gripper_open].
        :return: dict with keys matching PiZeroInference.forward().
        """
        # 1) Single-batch dimension
        bsz = 1
    
        # 2) Move/reshape image to (B,3,224,224) on the right device + dtype
        pixel_values = img_tensor.unsqueeze(0).to(device=self.device, dtype=self.dtype)
        # shape => (1, 3, 224, 224)
    
        # 3) Construct dummy text inputs (seq_length=10 for example)
        #    Typically the first 256 tokens denote "image tokens."
        seq_len = 276
        num_img_tokens = 256
        img_idx = getattr(self.model, "image_token_index", 257152)  # default if not found
        # Create a [B, 276] LongTensor 
        input_ids = torch.full(
            (bsz, seq_len),
            fill_value=2,  # e.g. some "text token" for the non-image portion
            dtype=torch.long,
            device=self.device,
        )
        # Overwrite the first 256 positions with the image_token_index => image tokens
        input_ids[:, :num_img_tokens] = img_idx

        # A full attention mask of 1's
        attention_mask = torch.ones((bsz, seq_len), dtype=torch.long, device=self.device)
    
        # 4) Build the PiZero causal masks + positions
        causal_mask, vlm_pos_ids, proprio_pos_ids, action_pos_ids = (
            self.model.build_causal_mask_and_position_ids(attention_mask, dtype=self.dtype)
        )
        image_text_proprio_mask, action_mask = self.model.split_full_mask_into_submasks(
            causal_mask
        )
    
        # 5) Convert the proprio to torch, shape => (B, cond_steps=1, proprio_dim=8)
        proprios = (
            torch.from_numpy(proprio_np)
            .float()
            .unsqueeze(0)  # B=1
            .unsqueeze(1)  # cond_steps=1
            .to(self.device, dtype=self.dtype)
        )  # => shape (1, 1, 8)
    
        # 6) Assemble the final dictionary for PiZero
        model_inputs = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_text_proprio_mask": image_text_proprio_mask.to(self.device),
            "action_mask": action_mask.to(self.device),
            "vlm_position_ids": vlm_pos_ids.to(self.device),
            "proprio_position_ids": proprio_pos_ids.to(self.device),
            "action_position_ids": action_pos_ids.to(self.device),
            "proprios": proprios,
        }
        return model_inputs

    def _send_joint_trajectory(self, target_joint_positions):
        """
        Example single-point trajectory. You must ensure your 'google_arm_controller'
        is a position-based trajectory controller that expects a point for each
        of your 11 joints in the exact same order as self.joint_names.
        """
        if len(target_joint_positions) != self.num_joints:
            logwarn(
                f"_send_joint_trajectory: mismatch: got {len(target_joint_positions)} actions vs {self.num_joints} joints"
            )
            return

        traj = JointTrajectory()
        traj.joint_names = self.joint_names

        pt = JointTrajectoryPoint()
        pt.positions = target_joint_positions.tolist()
        pt.time_from_start = rospy.Duration(1.0)  # 1 second to move
        traj.points.append(pt)

        goal = FollowJointTrajectoryGoal()
        goal.trajectory = traj

        self.client.send_goal(goal)
        # Optionally wait for result, or just send & forget
        # self.client.wait_for_result(rospy.Duration(2.0))

    def spin(self):
        rospy.spin()

def main():
    rospy.init_node("open_pizero_inference_node")
    node = GoogleRobotOpenPiZeroInferenceNode()
    node.spin()

if __name__ == "__main__":
    main()
