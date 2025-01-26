#!/usr/bin/env python3
import json
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
from geometry_msgs.msg import Pose
from std_srvs.srv import Trigger
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
import tf
import tf.transformations as tr

import hydra
from omegaconf import OmegaConf

# Import your PiZeroInference and any needed utilities
# Make sure pizero.py is on your PYTHONPATH or in the same directory
# e.g. from my_robot_inference.pizero import PiZeroInference
from src.model.vla.pizero import PiZeroInference
from src.agent.env_adapter.simpler import EDRSimplerAdapter

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

def logerr(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}][ERROR] {msg}")  

###############################
# Utility for loading checkpoint
###############################
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
    def __init__(self):
        # --- ROS Params ---
        self.loop_rate_hz = rospy.get_param("~loop_rate_hz", 2.0)
        self.checkpoint_path = rospy.get_param("~checkpoint_path", "/root/workspace/dataset/vla_log/2025-01-20_00-22_42_fractal_beta/checkpoint/step147900.pt")
        self.config_path = rospy.get_param("~config_path", "/root/workspace/open-pi-zero/config/eval/fractal_apple.yaml")
        self.gpu_id = rospy.get_param("~gpu_id", 0)
        self.use_bf16 = rospy.get_param("~use_bf16", True)
        self.use_torch_compile = rospy.get_param("~use_torch_compile", True)
        flow_sampling = rospy.get_param("~flow_sampling", "beta") # "beta" or "uniform"

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

        # Setup ROS subscriptions
        self.image_sub = rospy.Subscriber("/head_camera/image_raw", Image, self._image_callback, queue_size=1)
        self.joint_state_sub = rospy.Subscriber("/joint_states", JointState, self._joint_state_callback, queue_size=1)
        self.tf_listener = tf.TransformListener()

        # Setup ActionLib client for joint trajectory
        self.client = actionlib.SimpleActionClient("/google_fullbody_controller/follow_joint_trajectory",
                                                   FollowJointTrajectoryAction)
        loginfo("Waiting for google_fullbody_controller action server ...")
        self.client.wait_for_server()
        loginfo("Connected to google_fullbody_controller action server.")

        # Setup motion planner communication
        self.moveit_pose_pub = rospy.Publisher("/moveit_server/target_pose", Pose, queue_size=1, latch=True)
        loginfo("Waiting for /moveit_server/plan_arm service ...")
        rospy.wait_for_service("/moveit_server/plan_arm")
        self.moveit_plan_arm = rospy.ServiceProxy("/moveit_server/plan_arm", Trigger)
        loginfo("Connected to /moveit_server/plan_arm service")

        self._move_initial_pose()

        # We store the latest image and latest joint states
        self._clear_states()

        # Start a periodic timer to run inference + publish commands
        rospy.Timer(rospy.Duration(1.0 / self.loop_rate_hz), self._control_loop)

    def _image_callback(self, msg: Image):
        """
        Store the camera image. 
        Here, we'll keep it in HWC format so we can easily resize or process with OpenCV.
        """
        try:
            if self.latest_image is not None:
                return

            height, width = msg.height, msg.width
            channels = 3  # RGB8
            if msg.encoding == 'rgb8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, channels))
            else:
                raise Exception(f"image format {msg.encoding} is not supported yet")
            self.latest_image = img  # shape (H, W, 3) RGB8
        except Exception as e:
            logwarn(f"Failed to convert camera image: {e}")
            self.latest_image = None

    def _joint_state_callback(self, msg: JointState):
        """
        Extract relevant joint positions from /joint_states and build a dict
        """
        if self.latest_joints is not None:
            return

        name_to_pos = {}
        for i, nm in enumerate(msg.name):
            pos = msg.position[i]
            name_to_pos[nm] = pos

        self.latest_joints = name_to_pos

    def _prepare_model_inputs(self, eef_pos, eef_quat_xyzw, gripper_openness):
        """
        Manually build PiZero model inputs without calling simpler.py's 'preprocess()'.
        We'll do:
          - Image => resized to model's input size
          - Dummy text tokens & attention masks
          - Normalized 8D proprio if your model training expects that
        """
        # self.latest_image is shape (H,W,3) RGB
        # Resize to your model's input resolution
        resized_image = cv2.resize(
            self.latest_image,
            (self.cfg.env.adapter.image_size[0], self.cfg.env.adapter.image_size[1]),
            interpolation=cv2.INTER_AREA
        )
        # Convert to torch (B=1,3,H,W)
        images = torch.as_tensor(resized_image, dtype=torch.uint8).permute(2, 0, 1)[
            None
        ]  # [1, 3, H, W]
        instruction = "Pick a coke-can"
        model_inputs = self.adapter.processor(text=[instruction], images=images)

        # 3) Build normalized proprio (8D)
        #    If your model was trained with [-1..1] bounding, do the same here.
        gripper_closedness = 1.0 - gripper_openness # fractal in open-pi-zero requires closedness
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

    def _clear_states(self):
        self.latest_image = None
        self.latest_joints = None
        self.inference_busy = False

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

        # Check we have the latest camera image, finger joint states
        if self.latest_image is None or self.latest_joints is None:
            return

        # Mutex to prevent running inference multiple times
        if self.inference_busy:
            return
        self.inference_busy = True

        self.timer_count += 1
        start_t = time.time()

        # Gather EEF pose from TF
        try:
            # If your root frame is "world", and child is "link_gripper_tcp"
            now = rospy.Time(0)
            self.tf_listener.waitForTransform(
                "link_base", "link_gripper_tcp", now, rospy.Duration(0.5)
            )
            (trans, rot) = self.tf_listener.lookupTransform("link_base", "link_gripper_tcp", now)
            # trans => (x,y,z), rot => (qx,qy,qz,qw)
            eef_pos = np.array(trans, dtype=np.float32)
            eef_quat_xyzw  = np.array(rot, dtype=np.float32)
        except Exception as e:
            logwarn(f"Could not lookup TF for link_gripper_tcp: {e}")
            self._clear_states()
            return

        # Compute 'gripper_opening' from finger joints
        #    Assume 'joint_finger_left' and 'joint_finger_right' each in [0..1.3]
        left_finger = self.latest_joints.get("joint_finger_left", 0.0)
        right_finger = self.latest_joints.get("joint_finger_right", 0.0)
        gripper_max_range = 1.3
        # If 0 => open, 1.3 => closed, define 'gripper_opening' in [0..1]
        # Depending on how you used it in training, you might do the inverse
        # E.g. 0=closed, 1=open => adjust to match training.
        gripper_openness = 1.0 - ((left_finger + right_finger) / (2.0 * gripper_max_range))

        # Prepare the model inputs
        model_inputs = self._prepare_model_inputs(eef_pos, eef_quat_xyzw, gripper_openness)

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

        for eef_delta in raw_actions[: self.cfg.act_steps]: # in fractal, usually act_steps = 2
            # Convert from [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, Δgrip] into new EEF pose + new gripper value
            # For simplicity, let's parse them
            dx, dy, dz, ax, ay, az, gripper_action = eef_delta
            # Update position
            new_pos = eef_pos + np.array([dx, dy, dz], dtype=np.float32)
            # Convert axis-angle to a new orientation
            angle = np.linalg.norm([ax, ay, az])
            if angle < 1e-8:
                new_quat_xyzw = eef_quat_xyzw
            else:
                axis = np.array([ax, ay, az], dtype=np.float32) / angle
                delta_quat = tr.quaternion_about_axis(angle, axis)  # [x, y, z, w]
                new_quat_xyzw = tr.quaternion_multiply(eef_quat_xyzw, delta_quat)

            new_gripper_val = gripper_max_range * gripper_action # in fractal, 1 is closed = 1.3rad, 0 is opened = 0 rad
            new_joint_positions = self._moveit_ik_solver(new_pos, new_quat_xyzw, new_gripper_val)
            if new_joint_positions is None:
                logwarn("MoveIt IK solver failed. Skipping command.")
                self._clear_states()
                return
            break # for now, just run 1 step

        # Publish a JointTrajectory
        self._send_joint_trajectory(new_joint_positions)

        # Clear buffers so next loop waits for new sensor data
        self._clear_states()

    def moveit_server_plan_arm(self, pose_goal):
        # stub
        #joint_names =  ['joint_torso', 'joint_shoulder', 'joint_bicep', 'joint_elbow', 'joint_forearm', 'joint_wrist', 'joint_gripper']
        #positions = [0.0, 0.435, 0.0, 2.33, 0.0, -1.21, 0.0]
        #return (joint_names, positions)
        self.moveit_pose_pub.publish(pose_goal)

        try:
            response = self.moveit_plan_arm()
        except rospy.ServiceException as e:
            logerr(f"Service call failed: {e}")
            return None, None

        # 3. Check success/failure
        if not response.success:
            logerr(f"Planning failed: {response.message}")
            return None, None

        # 4. Parse the JSON result from response.message
        try:
            posture_dict = json.loads(response.message)
        except json.JSONDecodeError:
            logerr(f"Could not parse JSON from service response: {response.message}")
            return None, None

        # posture_dict should now be { "joint1_name": value, "joint2_name": value, ... }
        #loginfo("Planning succeeded. Final joint posture (parsed from JSON):")
        joint_names = list(posture_dict.keys())
        joint_positions = list(posture_dict.values())
        return (joint_names, joint_positions)


    def _moveit_ik_solver(self, eef_pos, eef_quat_xyzw, finger_val):
        """
        Use MoveIt to plan from the current state to the new EEF pose, then embed the finger joint.
        """
        pose_goal = Pose()
        pose_goal.position.x = float(eef_pos[0])
        pose_goal.position.y = float(eef_pos[1])
        pose_goal.position.z = float(eef_pos[2])
        pose_goal.orientation.x = float(eef_quat_xyzw[0])
        pose_goal.orientation.y = float(eef_quat_xyzw[1])
        pose_goal.orientation.z = float(eef_quat_xyzw[2])
        pose_goal.orientation.w = float(eef_quat_xyzw[3])

        joint_names, positions = self.moveit_server_plan_arm(pose_goal)

        if joint_names is None:
            return None

        plan_joint_map = {}
        for i, jn in enumerate(joint_names):
            plan_joint_map[jn] = positions[i]

        # Build a full array for all 11 joints in the correct order
        new_positions = []
        for jn in self.joint_names:
            if jn in plan_joint_map:
                new_positions.append(plan_joint_map[jn])
            else:
                curr_pos = self.latest_joints.get(jn, 0.0)
                new_positions.append(curr_pos)

        # Overwrite the finger joints
        if "joint_finger_left" in self.joint_names:
            idx_left = self.joint_names.index("joint_finger_left")
            new_positions[idx_left] = finger_val
        if "joint_finger_right" in self.joint_names:
            idx_right = self.joint_names.index("joint_finger_right")
            new_positions[idx_right] = finger_val

        # Overwrite the head joints (to look at the coke-can)
        if "joint_head_pan" in self.joint_names:
            idx_pan = self.joint_names.index("joint_head_pan")
            new_positions[idx_pan] = 0.0
        if "joint_head_tilt" in self.joint_names:
            idx_tilt = self.joint_names.index("joint_head_tilt")
            new_positions[idx_tilt] = 0.8

        return new_positions

    def _send_joint_trajectory(self, target_joint_positions):
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        pt = JointTrajectoryPoint()
        pt.positions = target_joint_positions
        pt.time_from_start = rospy.Duration(1.0)
        traj.points.append(pt)

        goal = FollowJointTrajectoryGoal()
        goal.trajectory = traj
        self.client.send_goal(goal)

    def _move_initial_pose(self):
        initial_pose = [0.0, 0.1432, 0.1476, 1.22, 0.0, 0.93, -1.394, 0.4, 0.4, 0.0, 0.8]
        self._send_joint_trajectory(initial_pose)

    def spin(self):
        rospy.spin()

def main():
    rospy.init_node("open_pizero_inference_node")
    node = GoogleRobotOpenPiZeroInferenceNode()
    node.spin()

if __name__ == "__main__":
    main()
