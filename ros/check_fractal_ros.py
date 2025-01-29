#!/usr/bin/env python3
import json
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.agent.dataset import TorchRLDSInterleavedDataset
from src.agent.env_adapter.base import BaseEnvAdapter

from rosif import RosIf

def main():
    config_path = "/workspaces/open-pi-zero/config/train/fractal.yaml"
    stats_path = "/workspaces/open-pi-zero/config/fractal_statistics.json"
    adapter = BaseEnvAdapter()

    # Load the exact same config YAML as used in your training script
    cfg = OmegaConf.load(config_path)
    with open(stats_path, "r") as f:
        dataset_statistics = json.load(f)

    rosif = RosIf()
    rosif.switch_controllers("trajectory")

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
    rosif.sleep_spin(3)

    # Get robot's posture after the motion in Gazebo, ROS
    (trans, rot) = rosif.lookup_transform("link_base", "link_gripper")

    # trans => (x,y,z), rot => (qx,qy,qz,qw)
    eef_pos = np.array(trans, dtype=np.float32)
    eef_quat_xyzw  = np.array(rot, dtype=np.float32)
    left_finger = rosif.latest_joints.get("joint_finger_left", 0.0)
    right_finger = rosif.latest_joints.get("joint_finger_right", 0.0)
    gripper_max_range = rosif.gripper_angle_closed
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

    user_input = input("press any key to start action validation...")
    rosif.switch_controllers("position")

    num_batches = 100
    for i in range(num_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            print("No more data in the loader.")
            break
    
        act = batch["action"]

        # Let's validate action side
        T = act.shape[1]  # number of time steps in this batch
        num_steps_to_replay = min(T, 5)

        # Loop over time steps for action evaluation
        for t in range(num_steps_to_replay):
            # Apply the dataset action a_t for 1 "time step"
            #     (i.e. parse the action, convert to EEF delta, do IK, etc.)
            norm_action_chunk = act[0, t].cpu().numpy()  # shape (4,7)
            for substep in range(norm_action_chunk.shape[0]):
                # e.g. shape (7,). If your dataset uses chunked actions, might be shape (horizon, 7)
                # This should be the same as PiZero model inference output
                norm_action_t = norm_action_chunk[substep]  # shape (7,)

                # Denormalize
                raw_action_except_gripper = adapter.denormalize_bound(
                    norm_action_t[:-1],  # first 6 dims
                    np.array(dataset_statistics["action"]["p01"])[:-1],
                    np.array(dataset_statistics["action"]["p99"])[:-1],
                    clip_min=-1, clip_max=1
                )
                # final dim is gripper, usually unnormalized
                raw_action_t = np.concatenate([raw_action_except_gripper, norm_action_t[-1:]], axis=0)
                # e.g. [dx, dy, dz, droll, dpitch, dyaw, dgrip]

                # Take pose delta
                dx, dy, dz = raw_action_t[0:3]
                # your Euler -> delta_quat approach:
                roll, pitch, yaw = raw_action_t[3:6]
                gripper_closedness = norm_action_t[-1]

                print(f"batch {i+1}, step {t+1}, sub {substep}, grp={gripper_closedness}")
                rosif.control_gripper(gripper_closedness)
                for l in range(300):# 3Hz
                    rosif.apply_action_with_servo(dx, dy, dz, roll, pitch, yaw)
                    rosif.sleep_spin(0.001) 

if __name__ == "__main__":
    main()
