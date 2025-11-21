#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from planners.gemini_planner import GeminiPlanner
import time
from std_msgs.msg import String, Empty
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Pose
import math
from tf.transformations import euler_from_quaternion

class DataRecorder:
    def __init__(self):
        self.global_map_data = None
        self.local_map_data = None
        self.global_map_required_flag = False
        self.local_map_required_flag = False
        self.cog_position = None
        self.cog_orientation = None
        self.cv_image = None
        self.true_dog_pose = Pose()
        self.true_uav_pose = Pose()
        self.bridge = CvBridge()
        self.gemini_planner = GeminiPlanner()
        self.init_ros()
        self.get_params()
        self.last_dog_pose_time = rospy.get_time()

    def init_ros(self):
        rospy.init_node('data_recorder')
        rospy.Subscriber('/llm/local_map_required', Empty, self.local_map_required)
        rospy.Subscriber('/llm/global_map_required', Empty, self.global_map_required)
        rospy.Subscriber('/llm/global_map', String, self.global_map)
        rospy.Subscriber('/llm/local_map', String, self.local_map)
        rospy.Subscriber('/quadrotor/uav/cog/odom', Odometry, self.cog_register, queue_size=1)
        rospy.Subscriber('/image_rect_color/compressed', CompressedImage, self.image_receiver, queue_size=1)
        rospy.Subscriber('/camera/image_raw/compressed', CompressedImage, self.image_receiver, queue_size=1)
        rospy.Subscriber('/quadrotor/mocap/pose', PoseStamped, self.true_uav_pose_cb, queue_size=1)
        rospy.Subscriber('/dog/mocap/pose', PoseStamped, self.true_dog_pose_cb, queue_size=1)
        rospy.Timer(rospy.Duration(1.0), self.check_dog_pose_timeout)

    def get_params(self):
        resolution = rospy.get_param('resolution', 1080)
        height = rospy.get_param('height', 2.3)
        FOV = rospy.get_param('FOV', 1.57)
        bias_x = rospy.get_param('bias_x', 1.0)
        bias_y = rospy.get_param('bias_y', 1.0)
        self.bias = [bias_x, bias_y]    
        hfgrid = 12 * resolution / 1080
        self.proportion = height/hfgrid * math.tan(FOV/2)

    def image_text_recorder(self, cv2_image_, response, cog_position=None, global_map_text=None, true_dog_pose=None, true_uav_pose=None):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        dir_name = time.strftime("/home/dragon_llm/ros/llm_ws/src/recorder/%Y%m%d")
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        image_filename_ = f"{dir_name}/withoutgrid_image_{timestamp}.jpeg"
        image_filename = f"{dir_name}/withgrid_image_{timestamp}.jpeg"
        text_filename = f"{dir_name}/withgrid_image_{timestamp}.txt"
        pose_filename = f"{dir_name}/withgrid_path_{timestamp}.txt"
        drone_position = str(cog_position.x-self.bias[0]) + ', ' + str(cog_position.y-self.bias[1])
        true_uav_position = str(true_uav_pose.position.x - self.bias[0]) + ', ' + str(true_uav_pose.position.y - self.bias[1])
        true_dog_position = str(true_dog_pose.position.x - self.bias[0]) + ', ' + str(true_dog_pose.position.y - self.bias[1])
        true_dog_ang = str(euler_from_quaternion([true_dog_pose.orientation.x, true_dog_pose.orientation.y, true_dog_pose.orientation.z, true_dog_pose.orientation.w])[2])
        if global_map_text is None:
            global_map_text = ''
        pose_text = drone_position + '\n' + true_uav_position + '\n' + true_dog_position + '\n' + true_dog_ang + '\n' + global_map_text
        cv2.imwrite(image_filename_, cv2_image_)
        cv2_image = self.gemini_planner.draw_elements(cv2_image_)
        cv2.imwrite(image_filename, cv2_image)
        with open(text_filename, 'w') as f:
            f.write(response)     
        with open(pose_filename, 'w') as f:
            f.write(pose_text)

    def true_uav_pose_cb(self, msg):
        self.true_uav_pose = msg.pose

    def true_dog_pose_cb(self, msg):
        self.true_dog_pose = msg.pose
        self.last_dog_pose_time = rospy.get_time()

    def check_dog_pose_timeout(self, event):
        if rospy.get_time() - self.last_dog_pose_time > 5.0:
            self.last_dog_pose_time = rospy.get_time()
            rospy.logwarn("No dog pose update for more than 5 seconds!")

    def cog_register(self, msg):
        self.cog_position = msg.pose.pose.position
        self.cog_orientation = msg.pose.pose.orientation

    def image_receiver(self, msg):
        self.cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
    
    def global_map_required(self, msg):
        self.global_map_required_flag = True

    def global_map(self, msg):
        self.global_map_data = msg.data
        self.global_map_required_flag = False

    def local_map_required(self, msg):
        self.local_map_required_flag = True
        self.cog_position_= self.cog_position
        self.cog_orientation_= self.cog_orientation
        self.cv_image_ = self.cv_image
        self.global_map_data_ = self.global_map_data
        self.true_dog_pose_ = self.true_dog_pose
        self.true_uav_pose_ = self.true_uav_pose
    
    def local_map(self, msg):
        self.local_map_data = msg.data
        self.local_map_required_flag = False
        self.image_text_recorder(self.cv_image_, msg.data, self.cog_position_, self.global_map_data_, self.true_dog_pose_, self.true_uav_pose_)

if __name__ == '__main__':
    datarecorder = DataRecorder()
    rospy.spin()
