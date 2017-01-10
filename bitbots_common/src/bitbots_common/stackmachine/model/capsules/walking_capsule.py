import copy
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class WalkingCapsule:

    def __init__(self):

        self.odometry_data = Odometry()
        self.pub_walking_objective: rospy.Publisher = None
        self.pub_walkin_params: rospy.Publisher = None

    def stop_walking(self):
        """ This method stops the walking - note that it could take some time until the robot is standing """
        raise NotImplementedError

    def is_walking(self):
        """ This method returns True if the walking is actually not running """
        raise NotImplementedError

    def start_walking_plain(self, f, s, tw):
        t = Twist()
        t.linear.x = f
        t.linear.y = s
        t.angular.z = tw
        self.pub_walkin_params.publish(t)

    def walking_callback(self, od: Odometry):
        self.odometry_data = copy.copy(od)