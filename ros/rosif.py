#!/usr/bin/env python3


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

import json
import rospy
import actionlib
from sensor_msgs.msg import JointState, Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from geometry_msgs.msg import Pose, TwistStamped
from std_srvs.srv import Trigger
import tf
import tf.transformations as tr

class RosIf():
    def __init__(self, node_name="open_pi_zero_rosif_node"):
        self.init(node_name)

    def init(self, node_name):
        rospy.init_node(node_name, anonymous=True)

        self.image_sub = rospy.Subscriber("/head_camera/image_raw", Image, self._image_callback, queue_size=1)
        self.joint_state_sub = rospy.Subscriber("/joint_states", JointState, self._joint_state_callback, queue_size=1)
        self.tf_listener = tf.TransformListener()

        rospy.loginfo("waiting for /google_fullbody_trajectory_controller/follow_joint_trajectory...")
        self.ros_control_fullbody = actionlib.SimpleActionClient("/google_fullbody_trajectory_controller/follow_joint_trajectory",
                                                   FollowJointTrajectoryAction)
        self.ros_control_gripper = actionlib.SimpleActionClient("/google_gripper_trajectory_controller/follow_joint_trajectory",
                                                   FollowJointTrajectoryAction)
        self.ros_control_fullbody.wait_for_server()
        self.ros_control_gripper.wait_for_server()

        self.moveit_pose_pub = rospy.Publisher("/moveit_server/planning_pose", Pose, queue_size=1, latch=True)
        rospy.loginfo("waiting for /moveit_server services...")
        rospy.wait_for_service("/moveit_server/plan_arm")
        rospy.wait_for_service("/moveit_server/switch_controller_position_group")
        rospy.wait_for_service("/moveit_server/switch_controller_trajectory")
        self.srv_moveit_plan_arm = rospy.ServiceProxy("/moveit_server/plan_arm", Trigger)
        self.srv_switch_controller_pos = rospy.ServiceProxy("/moveit_server/switch_controller_position_group", Trigger)
        self.srv_switch_controller_trj = rospy.ServiceProxy("/moveit_server/switch_controller_trajectory", Trigger)

        self.servo_pub = rospy.Publisher("/servo_server/delta_twist_cmds", TwistStamped, queue_size=1)

        self.gripper_angle_closed = 0.8
        self.gripper_angle_opened = 0.4

        self.clear_observation()
        rospy.loginfo("RosIf initialization done")

    def is_observation_available(self):
        return self.latest_image is not None and self.latest_joints is not None

    def clear_observation(self):
        self.latest_image = None
        self.latest_joints = None

    def _joint_state_callback(self, msg: JointState):
        name_to_pos = {}
        for i, nm in enumerate(msg.name):
            pos = msg.position[i]
            name_to_pos[nm] = pos
        self.latest_joints = name_to_pos

    def _image_callback(self, msg: Image):
        self.latest_image = msg

    def moveit_ik_solver(self, eef_pos, eef_quat_xyzw, finger_val):
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

    def move_initial_pose(self):
        joint_names = [
            "joint_torso", "joint_shoulder", "joint_bicep", "joint_elbow",
            "joint_forearm", "joint_wrist", "joint_gripper",
            "joint_finger_right", "joint_finger_left", "joint_head_pan", "joint_head_tilt"
        ]
        initial_pose = [0.0, 0.1432, 0.1476, 1.22, 0.0, 0.93, -1.394, 0.4, 0.4, 0.0, 0.8]
        joint_dict = dict(zip(joint_names, initial_pose))
        self.send_joint_trajectory(joint_dict)

    def _moveit_server_plan_arm(self, pose_goal):
        self.moveit_pose_pub.publish(pose_goal)

        try:
            response = self.srv_moveit_plan_arm()
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

    # return gripper's normalized state. 1 is opened, 0 is fully closed
    def get_gripper_closedness(self):
        left_finger = self.latest_joints.get("joint_finger_left", 0.0)
        right_finger = self.latest_joints.get("joint_finger_right", 0.0)
        gripper_max_range = self.gripper_angle_closed
        closeness = ((left_finger + right_finger) / (2.0 * gripper_max_range))
        openness = 1.0 - closeness
        return 1.0 if openness > 0.5 else 0.0

    def send_joint_trajectory(self, dict_target_joint_positions, controller='fullbody'):
        traj = JointTrajectory()
        traj.joint_names = dict_target_joint_positions.keys()
        pt = JointTrajectoryPoint()
        pt.positions = dict_target_joint_positions.values()
        pt.time_from_start = rospy.Duration(1.0)
        traj.points.append(pt)

        goal = FollowJointTrajectoryGoal()
        goal.trajectory = traj
        if controller == 'fullbody':
            self.ros_control_fullbody.send_goal(goal)
        elif controller == 'gripper':
            self.ros_control_gripper.send_goal(goal)
        else:
            rospy.logerr(f"Invalid controller type {controller}")

    # Suppose you have a policy action: dx, dy, dz, droll, dpitch, dyaw in local frame
    # and you want to apply it over dt=0.2 s at 10 Hz => 2 messages
    def apply_action_with_servo(self, dx, dy, dz, droll, dpitch, dyaw, time_scale=1.0):
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "link_base"

        msg.twist.linear.x = dx * time_scale
        msg.twist.linear.y = dy * time_scale
        msg.twist.linear.z = dz * time_scale
        msg.twist.angular.x = droll  * time_scale
        msg.twist.angular.y = dpitch * time_scale
        msg.twist.angular.z = dyaw   * time_scale

        self.servo_pub.publish(msg)

    def control_gripper(self, gripper_closeness):
        if gripper_closeness > 0.5:
            gripper_angle = self.gripper_angle_closed
        else:
            gripper_angle = self.gripper_angle_opened
        # calculate actual joint angle
        joint_dict = {}
        joint_dict["joint_finger_right"] = gripper_angle
        joint_dict["joint_finger_left"] = gripper_angle
        self.send_joint_trajectory(joint_dict, "gripper")

    def switch_controllers(self, mode):
        try:
            if mode == 'trajectory':
                response = self.srv_switch_controller_trj()
            elif mode == 'position':
                response = self.srv_switch_controller_pos()
            else:
                rospy.logerr("Invalid mode. Use 'trajectory' for trajectory controller or 'position' for position controller.")
                return
            if response.success:
                rospy.loginfo("Controllers switched successfully.")
            else:
                rospy.logerr("Failed to switch controllers.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
    
    def sleep_spin(self, duration_sec):
        duration_msec = int(duration_sec * 1000.0)
        rate = rospy.Rate(1000)
        for _ in range(duration_msec):
            try:
                rate.sleep()
            except rospy.exceptions.ROSTimeMovedBackwardsException:
                pass # if Gazebo world was reset, this happens
