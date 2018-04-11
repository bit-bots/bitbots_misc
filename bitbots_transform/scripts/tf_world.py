#!/usr/bin/env python2.7
import rospy
from sensor_msgs.msg import CameraInfo

from geometry_msgs.msg import TransformStamped
from gazebo_msgs.msg import ModelStates
import tf


class TFWorld(object):
    def __init__(self):
        rospy.init_node("world_transformer")
        rospy.Subscriber("/gazebo/model_states", ModelStates, self._callback, queue_size=1)
        rospy.spin()


    def _callback(self, msg):
        br = tf.TransformBroadcaster()
        for i in range(len(msg.name)):
            if msg.name[i] == "minibot":
                transform = TransformStamped()
                transform.header.frame_id = "world"
                transform.header.stamp = rospy.Time.now()
                transform.child_frame_id = "trunk_link"
                transform.transform.translation = msg.pose[i].position
                transform.transform.rotation = msg.pose[i].orientation
                br.sendTransformMessage(transform)
            elif msg.name[i] == "teensize_ball":
                transform = TransformStamped()
                transform.header.frame_id = "world"
                transform.header.stamp = rospy.Time.now()
                transform.child_frame_id = "ball"
                transform.transform.translation = msg.pose[i].position
                transform.transform.rotation = msg.pose[i].orientation
                br.sendTransformMessage(transform)

        transform = TransformStamped()
        transform.header.frame_id = "world"
        transform.header.stamp = rospy.Time.now()
        transform.child_frame_id = "base_link"
        transform.transform.translation = msg.pose[i].position
        transform.transform.translation.z = 0
        transform.transform.rotation = msg.pose[i].orientation
        br.sendTransformMessage(transform)


if __name__ == "__main__":
    TFWorld()