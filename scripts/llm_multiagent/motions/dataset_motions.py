import sys
import os
import json
import numpy as np
import rospy
import math
from std_msgs.msg import String, Empty, UInt8
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState
import tool
from .dog_motion import DogMotions

class DatasetMotions(DogMotions):
    def __init__(self, dataset_path, resolution=1080, height=2.3, FOV=1.57, bias=[1.0, 1.0]):
        # Initialize publishers/subscribers similar to DogMotions
        # But we might need to mock some of them or handle them differently
        
        self.dataset_path = dataset_path
        self.current_sample = None
        self.robot_position = np.array([0.0, 0.0]) # Virtual robot position
        self.robot_orientation = 0.0
        
        # Publishers for mocking
        self.pub_go1_vel = rospy.Publisher('/go1/cmd_vel', Twist, queue_size=10)
        self.local_map_required_pub = rospy.Publisher('/llm/local_map_required', Empty, queue_size=1)
        self.mission_target_pub = rospy.Publisher('/llm/mission_target', String, queue_size=1)
        self.rollback_pub = rospy.Publisher('/llm/rollback', Empty, queue_size=1)
        
        # Mock Odometry Publisher
        self.odom_pub = rospy.Publisher('/quadrotor/uav/cog/odom', Odometry, queue_size=10)
        
        # Subscriber for local map (we will publish to this ourselves in a loop or mock it)
        self.following_sub = None 
        
        # Internal state
        self.finish_flag = False
        self.mission_target = String()
        self.local_map_required = UInt8()
        
        # Parameters from DogMotions
        self.body_length = 2.2 * 2.7 / height
        self.threshold = 1.2 * 2.7 / height
        self.place_threshold = 1.0 * 2.3 / height
        self.carry_threshold = 2 * 2.3 / height
        self.bias = bias
        
        hfgrid = 12 * resolution / 1080
        self.proportion = height/hfgrid * math.tan(FOV/2)
        
        # Mock World Constructor Subscriber
        rospy.Subscriber('/llm/local_map_required', Empty, self.mock_perception)
        
        # Subscribe to cmd_vel to update virtual robot
        rospy.Subscriber('/go1/cmd_vel', Twist, self.update_virtual_robot)

        rospy.loginfo("DatasetMotions Initialized")

    def load_sample(self, sample_idx):
        # Load sample from dataset
        sample_file = os.path.join(self.dataset_path, f"sample_{sample_idx:06d}.json")
        if not os.path.exists(sample_file):
            rospy.logerr(f"Sample file not found: {sample_file}")
            return False
            
        with open(sample_file, 'r') as f:
            self.current_sample = json.load(f)
            
        # Reset robot state
        # In the dataset, the robot is usually at the center or start position?
        # The dataset has 'goal_position' in image coordinates [0,1].
        # We assume the robot starts at the center of the image (0.5, 0.5) which corresponds to (0,0) in our local frame?
        # Or we need to check 'start' if available. The new dataset JSON doesn't show 'start'.
        # Assuming robot is at center of the top-down view.
        self.robot_position = np.array([0.0, 0.0]) 
        self.robot_orientation = 0.0
        
        # Publish initial odometry
        self.publish_odom()
        return True

    def publish_odom(self):
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "odom"
        # Map robot position to 'cog' position expected by DogMotions
        # DogMotions: uav_cog = (cog_position.x - bias[0]) / proportion
        # So: cog_position.x = uav_cog * proportion + bias[0]
        
        odom.pose.pose.position.x = self.robot_position[0] * self.proportion + self.bias[0]
        odom.pose.pose.position.y = self.robot_position[1] * self.proportion + self.bias[1]
        
        # Orientation? DogMotions doesn't seem to use orientation from Odom for 'uav_cog', 
        # but it uses 'orientation' from local_map.
        
        self.odom_pub.publish(odom)
        
        # Also update self.uav_cog directly as DogMotions does in callback
        self.uav_cog = self.robot_position

    def update_virtual_robot(self, msg):
        # Simple integration of velocity
        dt = 0.1 # Assume 10Hz control loop or similar
        
        # Update orientation
        self.robot_orientation += msg.angular.z * dt
        
        # Update position
        # Velocity is in robot frame?
        # DogMotions sends Twist.
        # linear.x is forward velocity.
        
        dx = msg.linear.x * math.cos(self.robot_orientation) * dt
        dy = msg.linear.x * math.sin(self.robot_orientation) * dt
        
        self.robot_position[0] += dx
        self.robot_position[1] += dy
        
        self.publish_odom()

    def mock_perception(self, msg):
        # This replaces WorldConstructor
        if self.current_sample is None:
            return

        # Generate local map JSON based on current_sample and robot_position
        # The local map should contain:
        # - Main (Robot)
        # - Target
        # - Obstacles
        
        # Coordinate transformation:
        # Dataset [0,1] -> Local Frame (centered at image center)
        # 0.5, 0.5 -> 0, 0
        # Scale: 1920x1080 equivalent
        
        scale_x = 1920 / 80.0 # ~24 units
        scale_y = 1080 / 80.0 # ~13.5 units
        
        # Target Position (Global in the image frame)
        goal_norm = self.current_sample['metadata']['goal_position']
        target_x = (goal_norm[0] - 0.5) * scale_x
        target_y = (0.5 - goal_norm[1]) * scale_y # Y is usually inverted in images (top-down) vs standard cartesian
        
        # Robot Position (Global in the image frame)
        # self.robot_position is already in this frame (initialized at 0,0)
        robot_x = self.robot_position[0]
        robot_y = self.robot_position[1]
        theta = self.robot_orientation
        
        # Calculate parts for orientation
        # Body at center
        # Head forward (approx 1 unit)
        # Tail backward (approx 1 unit)
        head_x = robot_x + 1.0 * math.cos(theta)
        head_y = robot_y + 1.0 * math.sin(theta)
        tail_x = robot_x - 1.0 * math.cos(theta)
        tail_y = robot_y - 1.0 * math.sin(theta)
        
        # Construct JSON
        # Format from GeminiPlanner:
        # Objects: [{'name': '...', 'type': '...', 'coordinate': {'x': ..., 'y': ...}}, ...]
        
        objects = []
        
        # Robot (Main)
        objects.append({
            "name": "Robot_Dog",
            "type": "main",
            "coordinate": {"x": robot_x, "y": robot_y},
            "parts": [
                {"part": "body", "coordinate": {"x": robot_x, "y": robot_y}},
                {"part": "head", "coordinate": {"x": head_x, "y": head_y}},
                {"part": "tail", "coordinate": {"x": tail_x, "y": tail_y}}
            ]
        })
        
        # Target
        objects.append({
            "name": self.current_sample['metadata']['target_object'],
            "type": "target",
            "coordinate": {"x": target_x, "y": target_y}
        })
        
        # Obstacles?
        # We could parse mask, but for now let's assume empty or simple obstacles if needed.
        
        response_json = json.dumps(objects)
        
        # Publish to /llm/local_map
        # But DogMotions subscribes to it.
        # We can just call the callback directly if we want, but publishing is cleaner for ROS structure.
        # However, we are in the same process if we run this as a script.
        # Wait, DogMotions subscribes to /llm/local_map.
        
        pub = rospy.Publisher('/llm/local_map', String, queue_size=1)
        pub.publish(String(data=response_json))

    def go1_following_start(self, task):
        # Override to ensure we subscribe to the mock map
        self.following_sub = rospy.Subscriber('/llm/local_map', String, self.following)
        self.count = 0
        rospy.sleep(1) # Reduced sleep
        self.mission_target.data = task
        self.mission_target_pub.publish(self.mission_target)
        self.finish_flag = False
        self.local_map_required_pub.publish(Empty())
        self.start_point = self.uav_cog
        rospy.loginfo('DatasetMotions: Ready for following task')
        
        # We need a loop here or rely on callbacks.
        # DogMotions.go1_following_start has a while loop.
        
        while not self.finish_flag and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo('DatasetMotions: Following task finished')
        
        # Note: To use the actual Gemini VLM instead of Ground Truth, 
        # one would instantiate GeminiPlanner here, load the image from self.current_sample['image_path'],
        # and call gemini_planner.classifier_fast(task, image).
        # For baseline evaluation, we use Ground Truth to isolate navigation performance.

