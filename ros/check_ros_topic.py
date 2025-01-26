#!/usr/bin/env python3
# cd /root/workspace && uv run ./open-pi-zero/ros/check_ros_topic.py
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from std_msgs.msg import String

# ------------------------------------------------------------
# Workaround for rospy-all deserialization issue in Python 3.10
import codecs
def rosmsg_error_handler(exception):
    raise exception
codecs.register_error("rosmsg", rosmsg_error_handler)
# ------------------------------------------------------------

# Callbacks
def image_callback(msg: Image):
    print("/head_camera/image_raw received")
    #print(msg)

def joint_state_callback(msg: JointState):
    print("/joint_states received")
    #print(msg)

rospy.init_node('check_ros_node', anonymous=True)

# Subscriber
image_sub = rospy.Subscriber("/head_camera/image_raw", Image, image_callback, queue_size=1)
joint_state_sub = rospy.Subscriber("/joint_states", JointState, joint_state_callback, queue_size=1)

# Publisher
pub = rospy.Publisher('chatter', String, queue_size=10)

rate = rospy.Rate(1)
while not rospy.is_shutdown():
  hello_str = "hello world %s" % rospy.get_time()
  rospy.loginfo(hello_str)
  pub.publish(hello_str)
  rate.sleep()
