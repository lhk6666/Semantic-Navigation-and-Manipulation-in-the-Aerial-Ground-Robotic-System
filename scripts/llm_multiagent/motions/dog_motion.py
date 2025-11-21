import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from std_msgs.msg import UInt8, String, Empty, Float32, Float32MultiArray
from geometry_msgs.msg import Twist, Pose
from std_srvs.srv import Trigger
from gazebo_msgs.msg import ModelState
from nav_msgs.msg import Odometry
from gazebo_ros_link_attacher.srv import Attach, AttachRequest
import rospy
import math
import time
import json
import numpy as np
from planners import InitialVelocityPlanner, LLMPlanner
import tool
from MPC.mpc_planner_ import MPCPlanner

class DogMotions:
    def __init__(self, resolution=1080, height=2.3, FOV=1.57, bias = [1.0, 1.0]):
        self.pub_go1_vel = rospy.Publisher('/go1/cmd_vel', Twist, queue_size=10)
        self.pub_go1_pose = rospy.Publisher('/go1/body_pose', Pose, queue_size=10)
        self.service_client_sit = rospy.ServiceProxy('/go1/sit', Trigger)
        self.service_client_stand = rospy.ServiceProxy('/go1/stand', Trigger)
        self.local_map_required_pub = rospy.Publisher('/llm/local_map_required', Empty, queue_size=1)
        self.mission_target_pub = rospy.Publisher('/llm/mission_target', String, queue_size=1)
        self.go1_attach_pub = rospy.Publisher('/go1/attach', Empty, queue_size=1)
        self.go1_detach_pub = rospy.Publisher('/go1/detach', Empty, queue_size=1)
        self.rollback_pub = rospy.Publisher('/llm/rollback', Empty, queue_size=1)
        
        self.local_map_required = UInt8()
        self.mission_target = String()
        self.init_vel_planner = InitialVelocityPlanner()
        
        self.velocity_publishing_flag = False
        self.angular_velocity = 0
        self.finish_flag = False
        self.yolo_distance = [0, 0]
        self.yolo_distance_ = [0, 0]
        self.body_length = 2.2 * 2.7 / height
        self.threshold = 1.2 * 2.7 / height
        self.place_threshold = 1.0 * 2.3 / height
        self.carry_threshold = 2 * 2.3 / height
        self.bias = bias
        
        hfgrid = 12 * resolution / 1080
        self.proportion = height/hfgrid * math.tan(FOV/2)
        
        rospy.Subscriber('/quadrotor/uav/cog/odom', Odometry, self.cog_register)
        
    def cog_register(self, msg):
        cog_position = msg.pose.pose.position
        self.uav_cog = np.array([(cog_position.x - self.bias[0]) / self.proportion, (cog_position.y - self.bias[1]) / self.proportion])

    def go1_following_start(self, task):
        self.following_sub = rospy.Subscriber('/llm/local_map', String, self.following)
        self.count = 0
        rospy.sleep(4)
        self.mission_target.data = task
        self.mission_target_pub.publish(self.mission_target)
        self.finish_flag = False
        self.local_map_required_pub.publish(Empty())
        self.start_point = self.uav_cog
        rospy.loginfo('Dog: Ready for following task')
        while not self.finish_flag:
            rospy.sleep(0.1)
        rospy.loginfo('Dog: Following task finished')

    def following(self, msg):
        rospy.loginfo('Dog: Following')
        try:
            data = json.loads(msg.data)
            if tool.extract_coordinates_by_type(data, 'target'):
                target_position = np.array(tool.extract_coordinates_by_type(data, 'target'))
                following_flag = False
            else:
                target_position = self.uav_cog - self.start_point
                following_flag = True

            orientation = tool.extract_orientation_by_parts(data)
            main_positions, carry_flag, carry_success = tool.extract_coordinates_by_type(data, 'main', self.carry_threshold)
            if not main_positions:
                rospy.loginfo('Dog: I cannot find myself in the local map, I will wait for the next local map')
                self.local_map_required_pub.publish(Empty())
                self.start_point = self.uav_cog
                return
                
            if carry_flag and (not carry_success):
                self.rollback_pub.publish(Empty())
                self.detach_objects()
                self.following_sub.unregister()
                self.finish_flag = True
                rospy.loginfo('Dog: Orientation - ' + str(orientation) + ',\nMain Positions - ' + str(main_positions))
                return
            main_positions = np.array(main_positions)
            body_position = main_positions[0]   
            if (not carry_flag) or (carry_flag and len(main_positions) == 1):
                manipulator_position = body_position + self.body_length * np.array([math.cos(orientation), math.sin(orientation)])
                manipulator_position = round(manipulator_position[0], 1), round(manipulator_position[1], 1)
            else:
                manipulator_position = main_positions[1]
            obstacles = tool.extract_coordinates_by_type(data, 'obstacle')
            
            tar_vector = target_position - manipulator_position
            ori_vector = manipulator_position - body_position

            rospy.loginfo('Dog:\nBody Position - ' + str(body_position) + ',\Manipulator Position - ' + str(manipulator_position) + ',\nCurrent Orientation - ' + str(orientation) + ',\nTarget Position - ' + str(target_position) + ',\nObstacles Position - ' + str(obstacles) + ',\nDistance - ' + str(np.linalg.norm(target_position - manipulator_position)) + ',\nAlignment - ' + str(np.dot(tar_vector, ori_vector)/np.linalg.norm(ori_vector)/np.linalg.norm(tar_vector)))
            
            mag2tar_cos = np.dot(tar_vector, ori_vector)/np.linalg.norm(ori_vector)/np.linalg.norm(tar_vector)
            if (np.linalg.norm(target_position - manipulator_position) < self.threshold/abs(mag2tar_cos) and mag2tar_cos > 0.85 and not following_flag) or (carry_flag and (np.linalg.norm(target_position - manipulator_position) < self.place_threshold)):
                if carry_flag:
                    self.count += 1
                    if self.count >= 2:
                        self.following_sub.unregister()
                        self.finish_flag = True
                        return
                    else:
                        rospy.loginfo('Dog: It seems that I have reached the target, I will try again')
                        self.local_map_required_pub.publish(Empty())
                        self.start_point = self.uav_cog
                        return
                else:
                    self.following_sub.unregister()
                    self.finish_flag = True
                    return

            self.go1_cmd_semi_navi(body_position, orientation, target_position, obstacles, carry_flag)

            self.local_map_required_pub.publish(Empty())
            self.start_point = self.uav_cog

        except Exception as e:
            rospy.logerr(f"Dog: Error in following, I will try again - Reason: {e}\n")
            self.local_map_required_pub.publish(Empty())
            self.start_point = self.uav_cog

    def go1_cmd_semi_navi(self, body_position, orientation, target_position, obstacles, carry_flag, go1_ang=0.6, go1_vel=0.2, noncarry_compensation=1.3, carry_compensation=1.80, noncarryvel_compensation=0.85, carryvel_compensation=0.9):
        go1_twist = Twist()
        if carry_flag == False:
            target_orientation, _, chosen_rotation = self.init_vel_planner.compute_initial_velocity(body_position, target_position, obstacles, body_length=self.body_length + self.threshold, current_orientation=orientation)
        else:
            target_orientation, _, chosen_rotation = self.init_vel_planner.compute_initial_velocity(body_position, target_position, obstacles, body_length=self.body_length + self.threshold * 3, current_orientation=orientation)
        rospy.loginfo('Dog: Target Orientation - ' + str(target_orientation) + '\n' + 'Rotation - ' + str(chosen_rotation))
        if chosen_rotation == 'ccw':
            if orientation < 0:
                orientation += 2 * math.pi
            if target_orientation < 0:
                target_orientation += 2 * math.pi
            if orientation > target_orientation:
                target_orientation += 2 * math.pi
            orientation_diff = target_orientation - orientation
        elif chosen_rotation == 'cw':
            if orientation > 0:
                orientation -= 2 * math.pi
            if target_orientation > 0:
                target_orientation -= 2 * math.pi
            if orientation < target_orientation:
                target_orientation -= 2 * math.pi
            orientation_diff = target_orientation - orientation
        else:
            rospy.logwarn('Error formation in the local map, I will try again')
            return
        go1_ang = go1_ang if orientation_diff > 0 else -go1_ang
        motion_time = orientation_diff / go1_ang
        if not carry_flag:
            go1_twist.angular.z = go1_ang
            motion_time *= noncarry_compensation
        else:
            go1_twist.angular.z = go1_ang
            motion_time *= carry_compensation
        self.go1_cmd_twist_pub(go1_twist, abs(motion_time))

        go1_twist = Twist()
        manipulator_position = [body_position[0] + self.body_length * math.cos(target_orientation), body_position[1] + self.body_length * math.sin(target_orientation)]
        position_diff = [target_position[0] - body_position[0], target_position[1] - body_position[1]]
        orient_vector = [manipulator_position[0] - body_position[0], manipulator_position[1] - body_position[1]]
        if abs(np.dot(orient_vector, position_diff)/np.linalg.norm(orient_vector)/np.linalg.norm(position_diff)) < 0.85:
            distance_diff = np.linalg.norm(position_diff) * np.dot(orient_vector, position_diff) / np.linalg.norm(orient_vector) / np.linalg.norm(position_diff)
        elif abs(np.dot(orient_vector, position_diff)/np.linalg.norm(orient_vector)/np.linalg.norm(position_diff)) >= 0.85:
            distance_diff = np.dot(orient_vector, position_diff) / np.linalg.norm(orient_vector) - self.body_length - self.threshold * 0.5
        if distance_diff > 3:
            distance_diff = 3
        go1_vel = go1_vel if distance_diff > 0 else -go1_vel
        motion_time = distance_diff * self.proportion / go1_vel
        if np.linalg.norm(manipulator_position - target_position) > self.threshold * 0.8 or go1_vel < 0 or carry_flag:
            go1_vel = go1_vel
        elif np.linalg.norm(manipulator_position - target_position) <= self.threshold * 0.8 and go1_vel > 0 and not carry_flag:
            go1_vel = -go1_vel
            motion_time = 0.6
        if not carry_flag:
            go1_twist.linear.x = go1_vel
            motion_time *= noncarryvel_compensation
        else:
            go1_twist.linear.x = go1_vel
            motion_time *= carryvel_compensation
        self.go1_cmd_twist_pub(go1_twist, motion_time)

    def go1_cmd_twist_pub(self, go1_twist, motion_time=1):
        rate = rospy.Rate(100)
        start_time = time.time()
        self.pub_go1_vel.publish(go1_twist)
        while (not rospy.is_shutdown()) and time.time()-start_time<=motion_time:
            rate.sleep()
        self.pub_go1_vel.publish(Twist())

    def go1_attach_start(self, Letter = 'none'):
        self.catch_objects(Letter)
        rospy.sleep(0.3)
        go1_twist = Twist()
        go1_twist.linear.x = 0.2
        self.go1_cmd_twist_pub(go1_twist, motion_time=0.5)
        go1_twist.linear.x = 0
        go1_twist.angular.z = 0.4
        self.go1_cmd_twist_pub(go1_twist, motion_time=1.5)
        go1_twist.angular.z = -0.4
        self.go1_cmd_twist_pub(go1_twist, motion_time=3)
            
    def go1_detach_start(self, Letter = 'none'):
        self.detach_objects(Letter)

    def catch_objects(self, Letter = 'none'):
        self.go1_attach_pub.publish(Empty())

    def detach_objects(self, Letter = 'none'):
        self.go1_detach_pub.publish(Empty())

class DogMotionsSimulated(DogMotions):
    def __init__(self, resolution=1080, height=2.3, FOV=1.57, bias = [1.0, 1.0]):
        self.pub_go1_vel = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
        self.service_client_sit = rospy.ServiceProxy('/go1/sit', Trigger)
        self.service_client_stand = rospy.ServiceProxy('/go1/stand', Trigger)
        self.local_map_required_pub = rospy.Publisher('/llm/local_map_required', Empty, queue_size=1)
        self.mission_target_pub = rospy.Publisher('/llm/mission_target', String, queue_size=1)
        self.go1_attach_pub = rospy.Publisher('/go1/attach', Empty, queue_size=1)
        self.go1_detach_pub = rospy.Publisher('/go1/detach', Empty, queue_size=1)
        self.rollback_pub = rospy.Publisher('/llm/rollback', Empty, queue_size=1)
        
        self.local_map_required = UInt8()
        self.mission_target = String()
        self.init_vel_planner = InitialVelocityPlanner()
        
        self.velocity_publishing_flag = False
        self.angular_velocity = 0
        self.finish_flag = False
        self.yolo_distance = [0, 0]
        self.yolo_distance_ = [0, 0]
        self.body_length = 1.0 * 2.3 / height
        self.threshold = 1.1 * 2.7 / height
        self.place_threshold = 1.0 * 2.3 / height
        self.carry_threshold = 2 * 2.3 / height
        self.bias = bias
        
        hfgrid = 12 * resolution / 1080
        self.proportion = height/hfgrid * math.tan(FOV/2)
        
        rospy.Subscriber('/quadrotor/uav/cog/odom', Odometry, self.cog_register)

    def go1_cmd_twist_pub(self, go1_twist, motion_time=1):
        rospy.sleep(1)
        model_state = ModelState()
        model_state.model_name = "go1_gazebo"
        model_state.reference_frame = "base"
        model_state.twist.linear.x = go1_twist.linear.x * 1.65
        model_state.twist.linear.y = go1_twist.linear.y
        model_state.twist.linear.z = go1_twist.linear.z
        model_state.twist.angular.x = go1_twist.angular.x
        model_state.twist.angular.y = go1_twist.angular.y
        model_state.twist.angular.z = go1_twist.angular.z * 9
        self.pub_go1_vel.publish(model_state)
        start_time = time.time()
        while (not rospy.is_shutdown()) and time.time()-start_time<=motion_time:
            self.pub_go1_vel.publish(model_state)
            rospy.sleep(0.1)

    def catch_objects(self, Letter = 'none'):
        try:
            if Letter != 'none':
                Letter = Letter.lower()
                self.letter = Letter
            rospy.wait_for_service('/link_attacher_node/attach', timeout=2)
            attach_srv = rospy.ServiceProxy('/link_attacher_node/attach', Attach)
            req = AttachRequest()
            req.model_name_1 = 'go1_gazebo'
            req.link_name_1 = 'base'
            req.model_name_2 = Letter
            req.link_name_2 = Letter + '_link'
            resp = attach_srv(req)
            if resp.ok:
                rospy.loginfo("Successfully attached %s to %s", 'object', 'robot')
            else:
                rospy.logerr("Failed to attach %s to %s", 'object', 'robot')
        except Exception as e:
            pass

    def detach_objects(self, Letter = 'none'):
        try:
            if Letter != 'none':
                Letter = Letter.lower()
            else:
                Letter = self.letter
            rospy.wait_for_service('/link_attacher_node/detach', timeout=2)
            detach_srv = rospy.ServiceProxy('/link_attacher_node/detach', Attach)
            req = AttachRequest()
            req.model_name_1 = 'go1_gazebo'
            req.link_name_1 = 'base'
            req.model_name_2 = Letter
            req.link_name_2 = Letter + '_link'
            resp = detach_srv(req)
            if resp.ok:
                rospy.loginfo("Successfully detached %s from %s", 'object', 'robot')
            else:
                rospy.logerr(f"Failed to detach %s from %s", 'object', 'robot')
        except Exception as e:
            pass

class DogMotionsMPC(DogMotions):
    def __init__(self):
        self.go1_orinetation_sub = rospy.Subscriber('/yolo_detection/angle', Float32, self.get_orientation_state)
        self.go1_coordinate_sub = rospy.Subscriber('/yolo_detection/grid_coordinates', Float32MultiArray, self.get_head_coordinate)
        self.motion_planner = MPCPlanner(N=20, dt=0.1)
        self.kalmen_filter_ang = tool.KalmanFilter(1e-4, 1e-2)
        self.kalmen_filter_vel = tool.KalmanFilter(1e-4, 1e-2)

    def reset(self):
        self.go1_vel = Twist()
        self.pub_go1_vel.publish(self.go1_vel)
    
    def get_orientation_state(self,msg):
        self.go1_yaw = msg.data
        # print(self.go1_yaw)
        
    def get_head_coordinate(self, msg):
        self.yolo_distance[0] = round(msg.data[0],0)
        self.yolo_distance[1] = round(msg.data[1],0)
        self.yolo_distance_[0] = round(msg.data[0],1)
        self.yolo_distance_[1] = round(msg.data[1],1)
        # print (self.yolo_distance_)

    def sit(self):
        try:
            response = self.service_client_sit()
            if response.success:
                rospy.loginfo('Sit command executed successfully')
            else:
                rospy.logwarn('Stand command failed: %s', response.message)
        except rospy.ServiceException as e:
            rospy.logerr('Service call failed: %s', e)

    def stand(self):
        try:
            response = self.service_client_stand()
            if response.success:
                rospy.loginfo('Stand command executed successfully')
            else:
                rospy.logwarn('Stand command failed: %s', response.message)
        except rospy.ServiceException as e:
            rospy.logerr('Service call failed: %s', e)

    def go1_cmd_vel(self, x_velocity, y_velocity, motion_time=0.7):
        pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
        rate = rospy.Rate(1000) 
        model_state = ModelState()
        model_state.model_name = "go1_gazebo"
        model_state.reference_frame = "base"
        model_state.twist.linear.x = float(y_velocity)
        model_state.twist.linear.y = float(-x_velocity)
        model_state.twist.linear.z = 0
        model_state.twist.angular.x = 0
        model_state.twist.angular.y = 0
        model_state.twist.angular.z = self.angular_velocity
        start_time = time.time()
        self.velocity_publishing_flag = True
        while (not rospy.is_shutdown()) and time.time()-start_time<=motion_time:
            pub.publish(model_state)
            rate.sleep()
        self.velocity_publishing_flag = False
        
    def go1_cmd_vel_array(self, array, motion_time=3.5):
        pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
        rate = rospy.Rate(100) 
        model_state = ModelState()
        model_state.model_name = "go1_gazebo"
        model_state.reference_frame = "base"
        start_position = np.array([self.yolo_distance[0], self.yolo_distance[1]])
        for point in array:
            go1_vel = Twist()
            print(point)
            model_state.twist.angular.z = 0
            model_state.twist.linear.x = 0
            model_state.twist.linear.y = 0
            target_orientation = math.atan2(point[1], point[0]) - math.pi/2
            if target_orientation > math.pi:
                target_orientation -= 2 * math.pi
            elif target_orientation < -math.pi:
                target_orientation += 2 * math.pi
            dist_angle = target_orientation - self.go1_yaw
            if dist_angle > math.pi:
                dist_angle -= 2 * math.pi
            elif dist_angle < -math.pi:
                dist_angle += 2 * math.pi
            self.angular_velocity = dist_angle
            model_state.twist.angular.z = self.angular_velocity * 4
            start_time = time.time()
            self.velocity_publishing_flag = True
            go1_vel.angular.z = self.angular_velocity/3
            if go1_vel.angular.z > 0.06:
                go1_vel.angular.z = 0.2
            elif go1_vel.angular.z < -0.06:
                go1_vel.angular.z = -0.2
            self.pub_go1_vel.publish(go1_vel)
            pub.publish(model_state)
            while (not rospy.is_shutdown()):
                angle_diff = self.go1_yaw - target_orientation
                if angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                elif angle_diff < -math.pi:
                    angle_diff += 2 * math.pi
                if abs(angle_diff) < 0.05:
                    go1_vel.angular.z = 0
                    self.pub_go1_vel.publish(go1_vel)
                    break
                rate.sleep()
            model_state.twist.angular.z = 0
            model_state.twist.linear.x = self.proportion
            target_position = np.array(point) + start_position
            start_position = target_position
            dy = target_position[1] - self.yolo_distance_[1]
            dx = target_position[0] - self.yolo_distance_[0]
            yaw = self.go1_yaw + math.pi/2
            if yaw > math.pi:
                yaw -= 2 * math.pi
            elif yaw < -math.pi:
                yaw += 2 * math.pi
            go1_vel.linear.x = (dx * math.cos(yaw) + dy * math.sin(yaw)) * self.proportion
            go1_vel.linear.y = -(dx * math.sin(yaw) - dy * math.cos(yaw)) * self.proportion
            if 1/12 * math.pi < abs(math.atan2(go1_vel.linear.y, go1_vel.linear.x)) < 5/12 * math.pi or 7/12 * math.pi < abs(math.atan2(go1_vel.linear.y, go1_vel.linear.x)) < 11/12:
                temp = 0.1/min(abs(go1_vel.linear.x), abs(go1_vel.linear.y))
                go1_vel.linear.x *= temp
                go1_vel.linear.y *= temp
            else:
                if abs(go1_vel.linear.x) > abs(go1_vel.linear.y):
                    go1_vel.linear.x = 0.1 if go1_vel.linear.x > 0 else -0.1
                else:
                    go1_vel.linear.y = 0.1 if go1_vel.linear.y > 0 else -0.1
            self.pub_go1_vel.publish(go1_vel)
            while (not rospy.is_shutdown()):
                pub.publish(model_state)
                rate.sleep()
                if np.linalg.norm(np.array([self.yolo_distance_[0], self.yolo_distance_[1]]) - target_position) < 0.15:
                    go1_vel.linear.x = 0
                    go1_vel.linear.y = 0
                    self.pub_go1_vel.publish(go1_vel)
                    break
                elif np.linalg.norm(np.array([self.yolo_distance_[0], self.yolo_distance_[1]]) - target_position) > 2:
                    dy = target_position[1] - self.yolo_distance_[1]
                    dx = target_position[0] - self.yolo_distance_[0]
                    yaw = self.go1_yaw + math.pi/2
                    if yaw > math.pi:
                        yaw -= 2 * math.pi
                    elif yaw < -math.pi:
                        yaw += 2 * math.pi
                    go1_vel.linear.x = (dx * math.cos(yaw) + dy * math.sin(yaw)) * self.proportion
                    go1_vel.linear.y = -(dx * math.sin(yaw) - dy * math.cos(yaw)) * self.proportion
                    if 0.5 < abs(go1_vel.linear.x):
                        go1_vel.linear.x = 0.1 if go1_vel.linear.x > 0 else -0.1
                    if 0.5 < abs(go1_vel.linear.y):
                        go1_vel.linear.y = 0.1 if go1_vel.linear.y > 0 else -0.1
                    self.pub_go1_vel.publish(go1_vel)
        self.velocity_publishing_flag = False
    
    def go1_cmd_ang(self, current_orientation, target_orientation = 0, motion_time=3, forward_speed = 0):
        pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
        rate = rospy.Rate(1000) 
        model_state = ModelState()
        go1_vel = Twist()
        if target_orientation - current_orientation > math.pi:
            target_orientation -= 2 * math.pi
        elif target_orientation - current_orientation < -math.pi:
            target_orientation += 2 * math.pi
        model_state.model_name = "go1_gazebo"
        model_state.reference_frame = "base"
        if abs(target_orientation - current_orientation) > math.pi/2:
            model_state.twist.linear.x = -forward_speed
            go1_vel.linear.x = -forward_speed
        elif abs(target_orientation - current_orientation) < math.pi/2:
            model_state.twist.linear.x = forward_speed
            go1_vel.linear.x = forward_speed
        else:
            model_state.twist.linear.x = 0
            go1_vel.linear.x = 0
        print(go1_vel.linear.x)
        if abs(go1_vel.linear.x) < 0.2 and abs(go1_vel.linear.x) > 0:
            k_time = 0.2/abs(go1_vel.linear.x)
            if go1_vel.linear.x > 0:
                go1_vel.linear.x = 0.2
            else:
                go1_vel.linear.x = -0.2
        else:
            k_time = 1
        model_state.twist.linear.y = 0
        model_state.twist.linear.z = 0
        model_state.twist.angular.x = 0
        model_state.twist.angular.y = 0
        model_state.twist.angular.z = (target_orientation - current_orientation) /3 * k_time
        go1_vel.linear.y = 0
        go1_vel.linear.z = 0
        go1_vel.angular.x = 0
        go1_vel.angular.y = 0
        go1_vel.angular.z = (target_orientation - current_orientation) / 3 * k_time
        self.pub_go1_vel.publish(go1_vel)
        motion_time = motion_time / k_time
        start_time = time.time()
        while (not rospy.is_shutdown()) and time.time()-start_time<=motion_time:
            pub.publish(model_state)
            rate.sleep()
        go1_vel = Twist()
        go1_vel.linear.x = 0
        go1_vel.linear.y = 0
        go1_vel.linear.z = 0
        go1_vel.angular.x = 0
        go1_vel.angular.y = 0
        go1_vel.angular.z = 0
        self.pub_go1_vel.publish(go1_vel)
        
    def go1_cmd_navi(self, path, N = 20):
        go1_vel = Twist()
        path_x = np.array(path)[:,0]
        path_y = np.array(path)[:,1]
        path_theta = np.arctan2(np.gradient(path_y), np.gradient(path_x))
        dist = math.sqrt((path_x[-1] - path_x[0])**2 + (path_y[-1] - path_y[0])**2)
        k = dist / 2
        if k <= 1:
            k = 1
        if k > 1:
            k = 2
        self.motion_planner.reset()
        self.kalmen_filter_ang.reset()
        self.kalmen_filter_vel.reset()
        current_position = np.array([self.yolo_distance[0], self.yolo_distance[1], self.go1_yaw])
        for i in range(int(len(path)/k)):
            idx_ref = np.arange(i, min(i+N+1, len(path_x)))
            while len(idx_ref) < N+1:
                idx_ref = np.append(idx_ref, idx_ref[-1])
            ref_states = np.vstack([path_x[idx_ref],
                                    path_y[idx_ref],
                                    path_theta[idx_ref]])
            # with io.StringIO() as buf, redirect_stdout(buf):
            u = self.motion_planner.solve_mpc(current_position, ref_states)
            if 0.05 <= u[0] * self.proportion < 0.15:
                vel = 0.1
            elif -0.15 < u[0] * self.proportion <= -0.05:
                vel = -0.1
            elif -0.05 < u[0] * self.proportion < 0.05:
                vel = 0
            else:
                vel = u[0] * self.proportion
            if 0.06 <= u[1] < 0.18:
                ang = 0.12
            elif -0.18 < u[1] <= -0.05:
                ang = -0.12
            elif -0.06 < u[1] < 0.06:
                ang = 0
            else:
                ang = u[1]
            go1_vel.linear.x = vel
            go1_vel.angular.z = self.kalmen_filter_ang.update(ang) * 1.17 
            self.pub_go1_vel.publish(go1_vel)
            new_x = current_position[0] + vel/self.proportion*0.1*np.cos(current_position[2])
            new_y = current_position[1] + vel/self.proportion*0.1*np.sin(current_position[2])
            new_theta = current_position[2] +ang*0.1
            current_position = np.array([new_x, new_y, new_theta])
            rospy.sleep(0.09)
        print('Current position - Expect: ' + str(current_position), 'Real: ' + str([self.yolo_distance[0], self.yolo_distance[1], self.go1_yaw]))
        go1_vel.linear.x = 0
        go1_vel.angular.z = 0
        self.pub_go1_vel.publish(go1_vel)

    def rotating(self, msg):
        rospy.loginfo('Dog: Rotating')
        try:
            print(msg.data)
            data = json.loads(msg.data)
            current_orientation = int(data["current orientation"])
            target_orientation = int(data["target orientation"])
            # print(distance)
            self.go1_cmd_ang(current_orientation, target_orientation)
            # print(distance)
            if current_orientation == target_orientation:
                self.rotating_sub.unregister()
                self.finish_flag = True
            else:
                self.count = 0
        except Exception as e:
            rospy.logerr(f"Error in rotating: {e}")

class DogMotionsLLM(DogMotionsMPC):
    def __init__(self):
        self.llm_planner = LLMPlanner()
    
    def following(self, msg):
        rospy.loginfo('Dog: Following')
        try:
            # print(msg.data)
            data = json.loads(msg.data)
            
            data["object position"] = self.yolo_distance
            rospy.loginfo(data)
            
            self.llm_planner.init(data)
            
            if self.yolo_distance == []:
                obj_distance = np.array(data["object position"])
            else:   
                obj_distance = np.array(self.yolo_distance)
            if (abs(obj_distance[0]) <= 0.5 and abs(obj_distance[1]) <= 0.5) and (data["target position"] is None):
                self.count += 1
                if self.count > 2:
                    self.following_sub.unregister()
                    self.finish_flag = True
                return
            elif data["target position"] is not None:
                if math.sqrt((obj_distance[0] - data["target position"][0])**2 + (obj_distance[1] - data["target position"][1])**2) < 1:
                    self.following_sub.unregister()
                    self.finish_flag = True
                    return
            # self.count = 0
            # if data["obstacles position"] == None:
            #     # print(distance)
            #     if data["target position"] is not None:
            #         target = data["target position"]
            #         # target_orientation = math.atan2(target[1] - obj_distance[1], target[0] - obj_distance[0]) - math.pi/2
            #         # if target_orientation > math.pi:
            #         #     target_orientation -= 2 * math.pi
            #         # elif target_orientation < -math.pi:
            #         #     target_orientation += 2 * math.pi
            #         # print(str(self.go1_yaw) + '\n' + str(target_orientation) + '\n' + str(self.proportion * np.linalg.norm(target - obj_distance)/6) + '\n' )
            #         # self.go1_cmd_ang(self.go1_yaw, target_orientation, forward_speed = self.proportion * np.linalg.norm(target - obj_distance)/6)
            #         path,vel = cbt.generate_safe_path((obj_distance[0], obj_distance[1]), (target[0], target[1]), [], obstacle_effect_area=1.75, num_points=50)
            #     elif data["target position"] is None:
            #         # target_orientation = math.atan2(-obj_distance[1], -obj_distance[0]) - math.pi/2
            #         # if target_orientation > math.pi:
            #         #     target_orientation -= 2 * math.pi
            #         # elif target_orientation < -math.pi:
            #         #     target_orientation += 2 * math.pi
            #         # print(str(target_orientation) + '\n' + str(self.proportion * np.linalg.norm(obj_distance)/6) + '\n' )
            #         # self.go1_cmd_ang(self.go1_yaw, target_orientation, forward_speed= self.proportion * np.linalg.norm(obj_distance)/6)
            #         path,vel = cbt.generate_safe_path((obj_distance[0], obj_distance[1]), (0, 0), [], obstacle_effect_area=1.75, num_points=50)
            # else:
            #     obstacles = data["obstacles position"]
            #     if data["target position"] is None:
            #         path,vel = cbt.generate_safe_path((obj_distance[0], obj_distance[1]), (0, 0.0), obstacles, obstacle_effect_area=1.75, num_points=50)
            #         # vel = self.init_vel_planner.compute_initial_velocity((obj_distance[0], obj_distance[1]), (0, 0.0), obstacles)
            #     elif data["target position"] is not None:
            #         target = data["target position"]
            #         path,vel = cbt.generate_safe_path((obj_distance[0], obj_distance[1]), (target[0], target[1]), obstacles, obstacle_effect_area=1.75, num_points=50)
            #         # vel = self.init_vel_planner.compute_initial_velocity((obj_distance[0], obj_distance[1]), (target[0], target[1]), obstacles)
            # self.go1_cmd_navi(path)
            
            posi_array = self.llm_planner.run()
            vel_array = []
            for i in range(1, len(posi_array)):
                dx = posi_array[i][0] - posi_array[i-1][0]
                dy = posi_array[i][1] - posi_array[i-1][1]
                vel_array.append([dx, dy])
            self.go1_cmd_vel_array(vel_array)
            
            self.local_map_required_pub.publish(self.local_map_required)
            # print(distance)
        except Exception as e:
            rospy.logerr(f"Error in following: {e}")
