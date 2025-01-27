#!/usr/bin/env python3
import json
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.agent.dataset import TorchRLDSInterleavedDataset
from src.agent.env_adapter.base import BaseEnvAdapter


# Workaround for rospy-all deserialization issue in Python 3.10
import codecs
def rosmsg_error_handler(exception):
    raise exception
codecs.register_error("rosmsg", rosmsg_error_handler)

import rospy
import actionlib
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from geometry_msgs.msg import Pose
from std_srvs.srv import Trigger
import tf
import tf.transformations as tr
class RosIf():
    def __init__(self):
        self.init()

    def init(self):
        rospy.init_node("check_fractal_ros", anonymous=True)
        self.joint_state_sub = rospy.Subscriber("/joint_states", JointState, self._joint_state_callback, queue_size=1)
        self.tf_listener = tf.TransformListener()

        self.ros_control_client = actionlib.SimpleActionClient("/google_fullbody_controller/follow_joint_trajectory",
                                                   FollowJointTrajectoryAction)
        self.ros_control_client.wait_for_server()

        self.moveit_pose_pub = rospy.Publisher("/moveit_server/target_pose", Pose, queue_size=1, latch=True)
        rospy.wait_for_service("/moveit_server/plan_arm")

        self.moveit_plan_arm = rospy.ServiceProxy("/moveit_server/plan_arm", Trigger)

    def _joint_state_callback(self, msg: JointState):
        name_to_pos = {}
        for i, nm in enumerate(msg.name):
            pos = msg.position[i]
            name_to_pos[nm] = pos
        self.latest_joints = name_to_pos

    def moveit_ik_solver(self, eef_pos, eef_quat_xyzw, finger_val):
        joint_names = [
            "joint_torso", "joint_shoulder", "joint_bicep", "joint_elbow",
            "joint_forearm", "joint_wrist", "joint_gripper",
            "joint_finger_right", "joint_finger_left", "joint_head_pan", "joint_head_tilt"
        ]
        pose_goal = Pose()
        pose_goal.position.x = float(eef_pos[0])
        pose_goal.position.y = float(eef_pos[1])
        pose_goal.position.z = float(eef_pos[2])
        pose_goal.orientation.x = float(eef_quat_xyzw[0])
        pose_goal.orientation.y = float(eef_quat_xyzw[1])
        pose_goal.orientation.z = float(eef_quat_xyzw[2])
        pose_goal.orientation.w = float(eef_quat_xyzw[3])
        joint_names, positions = self._moveit_server_plan_arm(pose_goal)
        if joint_names is None:
            return None
        plan_joint_map = {}
        for i, jn in enumerate(joint_names):
            plan_joint_map[jn] = positions[i]
        # Build a full array for all 11 joints in the correct order
        new_positions = []
        for jn in joint_names:
            if jn in plan_joint_map:
                new_positions.append(plan_joint_map[jn])
            else:
                curr_pos = self.latest_joints.get(jn, 0.0)
                new_positions.append(curr_pos)

        # Overwrite the finger joints
        if "joint_finger_left" in joint_names:
            idx_left = joint_names.index("joint_finger_left")
            new_positions[idx_left] = finger_val
        if "joint_finger_right" in joint_names:
            idx_right = joint_names.index("joint_finger_right")
            new_positions[idx_right] = finger_val

        # return dict of joint names and positions
        dict_joint_positions = dict(zip(joint_names, new_positions))
        return dict_joint_positions

    def _moveit_server_plan_arm(self, pose_goal):
        self.moveit_pose_pub.publish(pose_goal)

        try:
            response = self.moveit_plan_arm()
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return None, None

        if not response.success:
            rospy.logerr(f"Planning failed: {response.message}")
            return None, None

        try:
            posture_dict = json.loads(response.message)
        except json.JSONDecodeError:
            rospy.logerr(f"Could not parse JSON from service response: {response.message}")
            return None, None

        # posture_dict should now be { "joint1_name": value, "joint2_name": value, ... }
        joint_names = list(posture_dict.keys())
        joint_positions = list(posture_dict.values())
        return (joint_names, joint_positions)

    def lookup_transform(self, target_frame, source_frame):
        now = rospy.Time(0)
        self.tf_listener.waitForTransform(
        target_frame, source_frame, now, rospy.Duration(0.5)
        )
        (trans, rot) = self.tf_listener.lookupTransform(target_frame, source_frame, now)
        return (trans, rot)
    
    def send_joint_trajectory(self, dict_target_joint_positions):
        traj = JointTrajectory()
        traj.joint_names = dict_target_joint_positions.keys()
        pt = JointTrajectoryPoint()
        pt.positions = dict_target_joint_positions.values()
        pt.time_from_start = rospy.Duration(1.0)
        traj.points.append(pt)

        goal = FollowJointTrajectoryGoal()
        goal.trajectory = traj
        self.ros_control_client.send_goal(goal)
    
    def sleep_spin(self, duration_sec):
        rate = rospy.Rate(10)
        for _ in range(duration_sec * 10):
            rate.sleep()

def main():
    config_path = "/workspaces/open-pi-zero/config/train/fractal.yaml"
    stats_path = "/workspaces/open-pi-zero/config/fractal_statistics.json"
    adapter = BaseEnvAdapter()

    # Load the exact same config YAML as used in your training script
    cfg = OmegaConf.load(config_path)
    with open(stats_path, "r") as f:
        dataset_statistics = json.load(f)

    rosif = RosIf()

    # Initialize dataset, note that dataset will be normalized while loading
    dataset_obj = TorchRLDSInterleavedDataset(cfg.data.train, train=False)
    dataset = dataset_obj.dataset
    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    data_iter = iter(loader)
    batch = next(data_iter)

    # Inspect the keys
    print("=== Keys in batch ===")
    print(list(batch.keys()))
    # Typically you’ll see: ['observation', 'action', 'task', 'dataset_name', 'action_pad_mask']

    # The main things you want for your “round‐trip” check are the
    # `observation` (which has proprio + images) and `action`.
    obs = batch["observation"]
    act = batch["action"]
    txt = batch["task"]["language_instruction"]  # usually shape [B], with 1 element

    print("=== Observations ===")
    for k, v in obs.items():
        try:
            print(f"{k}: shape={list(v.shape)}, dtype={v.dtype}")
        except:
            print(f"{k}: -- no shape --")

    # 6) Extract the proprio, which is typically shape [B, window_size, some_dim]
    norm_proprio = obs["proprio"][0, 0].cpu().numpy()
    # If you also want the raw image frames:
    #images = obs["image_primary"]  # shape [B=1, T=1, H, W, C]

    # Since proprio in the data is already normalized while loading by TorchRLDSInterleavedDataset,
    # We denormalized proprio first to get the actual eef pose
    denorm_proprio = adapter.denormalize_bound(
        norm_proprio,
        np.array(dataset_statistics["proprio"]["p01"]),
        np.array(dataset_statistics["proprio"]["p99"]),
        clip_min=-1,
        clip_max=1,
    )

    # denormalized proprio to pos, quotanion, gripper joint
    proprio_pos = denorm_proprio[0:3]
    proprio_quat_xyzw = denorm_proprio[3:7]
    proprio_gripper_closeness = denorm_proprio[7]

    dict_new_joint_positions = rosif.moveit_ik_solver(proprio_pos, proprio_quat_xyzw, proprio_gripper_closeness)
    if dict_new_joint_positions is None:
        print("ERROR: Motion planning failed")
        return
    rosif.send_joint_trajectory(dict_new_joint_positions)
    rosif.sleep_spin(5)

    # Get robot's posture after the motion in Gazebo, ROS
    (trans, rot) = rosif.lookup_transform("link_base", "link_gripper")

    # trans => (x,y,z), rot => (qx,qy,qz,qw)
    eef_pos = np.array(trans, dtype=np.float32)
    eef_quat_xyzw  = np.array(rot, dtype=np.float32)
    left_finger = rosif.latest_joints.get("joint_finger_left", 0.0)
    right_finger = rosif.latest_joints.get("joint_finger_right", 0.0)
    gripper_max_range = 1.3
    gripper_closedness = ((left_finger + right_finger) / (2.0 * gripper_max_range))
    raw_proprio_ros = np.concatenate([eef_pos, eef_quat_xyzw, [gripper_closedness]], axis=0)  # shape (3+4+1=8,)

    # Normalized measured proprio in Gazebo, ROS
    norm_proprio_ros = adapter.normalize_bound(
        raw_proprio_ros,
        np.array(dataset_statistics["proprio"]["p01"]),
        np.array(dataset_statistics["proprio"]["p99"]),
        clip_min=-1,
        clip_max=1,
    )

    # Now let's see how the dataset is presenting the raw proprio.
    # We can try normalizing it with the same stats:
    print(f"Normalized proprio in the dataset:\n{norm_proprio}")
    print(f"Normalized proprio in the ROS:\n{norm_proprio_ros}")
    dist = norm_proprio - norm_proprio_ros
    print(f"Distance:\n{dist}")

    print(f"Denormalized proprio in the dataset:\n{denorm_proprio}")
    print(f"Measured proprio in the ROS:\n{raw_proprio_ros}")
    dist = denorm_proprio - raw_proprio_ros
    print(f"Distance:\n{dist}")

if __name__ == "__main__":
    main()
