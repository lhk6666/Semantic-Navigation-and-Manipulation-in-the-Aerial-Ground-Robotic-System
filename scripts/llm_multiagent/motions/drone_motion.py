import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from std_msgs.msg import Empty, String, UInt8
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from ros_chatgpt.msg import FlightNav
from tf.transformations import quaternion_from_euler
import rospy
import threading
import numpy as np
import math
import json
from planners import PathPlanner
import tool
from pathlib import Path
import yaml

class DroneMotions:
    def __init__(self, resolution = 1080, arg_fix_posi = 'drone', height = 2.3, FOV = 1.57, bias = [1.0, 1.0]):
        start_pub = rospy.Publisher('quadrotor/teleop_command/start', Empty, queue_size=1)
        self.nav_pub = rospy.Publisher('quadrotor/uav/nav', FlightNav, queue_size=1)
        self.takeoff_pub = rospy.Publisher('quadrotor/teleop_command/takeoff', Empty, queue_size=1)
        self.image_signal_pub = rospy.Publisher('quadrotor/teleop_command/uav/image_signal', Empty, queue_size=1)
        self.posi_pub = rospy.Publisher('/quadrotor/target_pose', PoseStamped, queue_size=1)
        self.shoot_pub = rospy.Publisher('/llm/shoot', String, queue_size=1)
        self.global_map_required_pub = rospy.Publisher('/llm/global_map_required', Empty, queue_size=1)
        self.mission_target_pub = rospy.Publisher('/llm/mission_target', String, queue_size=1)
        rospy.sleep(0.5)
        start_pub.publish(Empty())
        rospy.sleep(0.5)

        self.stop_flag = True
        self.lock = threading.Lock()
        self.ready = False
        self.construct_flag = False
        self.wait_dog_flag = False
        self.drone_back_flag = False
        self.rollback_flag = False
        self.dog_update_flag = False
        self.shoot_msg = String()
        self.path_planner = PathPlanner(num_ctrl_points=8, obstacle_effect_area=3)
        
        
        self.cog_sub = rospy.Subscriber('/quadrotor/uav/cog/odom', Odometry, self.cog_register, queue_size=1)
        self.state_sub = rospy.Subscriber('/quadrotor/flight_state', UInt8, self.state_register, queue_size=1)
        self.rollback_sub = rospy.Subscriber('/llm/rollback', Empty, self.rollback, queue_size=1)
        self.local_map_sub = rospy.Subscriber('/llm/local_map', String, self.local_map_register, queue_size=1)
        
        hf_grid = 12 * resolution / 1080
        self.grid = hf_grid
        self.proportion = height/hf_grid * math.tan(FOV/2)
        self.height = height
        self.bias = bias
        
        if arg_fix_posi == 'drone':
            self.fix_flag = False
        else:
            self.fix_flag = True
            
        cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

    def rollback(self, msg):
        self.rollback_flag = True

    def local_map_register(self, msg):
        try:
            data = json.loads(msg.data)
            main_position, _, _ = tool.extract_coordinates_by_type(data, 'main')
            if not main_position:
                self.drone_back_flag = True
            else:
                body_position = main_position[0]
                self.dog_position = np.array(body_position)
                self.dog_update_flag = True
                self.drone_back_flag = False
        except Exception as e:
            rospy.logerr(f"Drone: Error in local_map_register: {e}")
        
    def state_register(self, msg):
        if msg.data == 5:
            self.ready = True
        
    def cog_register(self, msg):
        self.cog_position = msg.pose.pose.position
        self.cog_orientation = msg.pose.pose.orientation
        self.cog_linear = msg.twist.twist.linear
        self.cog_angular = msg.twist.twist.angular

    def quad_cmd_vel(self, x_velocity, y_velocity, z_velocity=0):
        nav_msg = FlightNav()
        nav_msg.control_frame = FlightNav.WORLD_FRAME
        nav_msg.target = FlightNav.COG
        nav_msg.pos_z_nav_mode = FlightNav.VEL_MODE
        nav_msg.pos_xy_nav_mode = FlightNav.VEL_MODE
        nav_msg.target_vel_x = y_velocity 
        nav_msg.target_vel_y = -x_velocity
        nav_msg.target_vel_z = z_velocity
        self.nav_pub.publish(nav_msg)   
        
    def quad_cmd_ang(self, z_angular=0):
        nav_msg = FlightNav()
        nav_msg.control_frame = FlightNav.WORLD_FRAME
        nav_msg.target = FlightNav.COG
        nav_msg.pos_z_nav_mode = FlightNav.VEL_MODE
        nav_msg.pos_xy_nav_mode = FlightNav.VEL_MODE
        nav_msg.target_yaw = z_angular
        self.nav_pub.publish(nav_msg)   

    def quad_takeoff(self):
        i = 0
        self.takeoff_pub.publish(Empty())
        while not self.ready:
            continue       
        self.quad_height(self.height)
        self.image_signal_pub.publish(Empty())
        
    def quad_navi(self, array, angular = math.pi * 22/45, wait=True):
        i = 0
        posi_msg = PoseStamped()
        if array[0] >= 15 or array[0] <= -15 or array[1] >= 15 or array[1] <= -15:
            rospy.logerr("Drone: Target position out of range, recalculate path")
            current_position = ((self.cog_position.x - self.bias[0]) / self.proportion, (self.cog_position.y - self.bias[1]) / self.proportion)
            distance = np.linalg.norm(np.array(current_position) - np.array(self.target))
            num_points = max(30, int(distance * 10))  # Adjust num_points based on distance
            self.path = self.path_planner.generate_safe_path(current_position, self.target, self.obstacles)
            return
        posi_msg.pose.position.x = array[0] + self.bias[0]
        posi_msg.pose.position.y = array[1] + self.bias[1]
        quaternion = quaternion_from_euler(0, 0, angular)
        posi_msg.pose.orientation.x = quaternion[0]
        posi_msg.pose.orientation.y = quaternion[1]
        posi_msg.pose.orientation.z = quaternion[2]
        posi_msg.pose.orientation.w = quaternion[3]
        posi_msg.pose.position.z = self.height
        self.posi_pub.publish(posi_msg)
        while True and wait:
            try:
                position_change = np.sqrt(
                    (self.cog_position.x - prev_cog_position.x) ** 2 +
                    (self.cog_position.y - prev_cog_position.y) ** 2 +
                    (self.cog_position.z - prev_cog_position.z) ** 2
                )
                orientation_change = np.sqrt(
                (self.cog_orientation.x - prev_cog_orientation.x) ** 2 +
                (self.cog_orientation.y - prev_cog_orientation.y) ** 2 +
                (self.cog_orientation.z - prev_cog_orientation.z) ** 2 +
                (self.cog_orientation.w - prev_cog_orientation.w) ** 2
                )
            except:
                prev_cog_position = self.cog_position
                prev_cog_orientation = self.cog_orientation
                continue
            
            if position_change < 0.01 and orientation_change < 0.01:
                i += 1
                if i > 3:    
                    break
                
            prev_cog_position = self.cog_position
            prev_cog_orientation = self.cog_orientation
            rospy.sleep(0.3)
        if self.construct_flag:
            self.shoot()
        
    def quad_height(self, height, angular = math.pi * 22/45):
        i = 0
        posi_msg = PoseStamped()
        posi_msg.pose.position.x = self.cog_position.x
        posi_msg.pose.position.y = self.cog_position.y
        posi_msg.pose.position.z = height
        quaternion = quaternion_from_euler(0, 0, angular)
        posi_msg.pose.orientation.x = quaternion[0]
        posi_msg.pose.orientation.y = quaternion[1]
        posi_msg.pose.orientation.z = quaternion[2]
        posi_msg.pose.orientation.w = quaternion[3]
        self.posi_pub.publish(posi_msg)
        while True:
            try:
                position_change = np.sqrt(
                    (self.cog_position.x - prev_cog_position.x) ** 2 +
                    (self.cog_position.y - prev_cog_position.y) ** 2 +
                    (self.cog_position.z - prev_cog_position.z) ** 2
                )
                orientation_change = np.sqrt(
                (self.cog_orientation.x - prev_cog_orientation.x) ** 2 +
                (self.cog_orientation.y - prev_cog_orientation.y) ** 2 +
                (self.cog_orientation.z - prev_cog_orientation.z) ** 2 +
                (self.cog_orientation.w - prev_cog_orientation.w) ** 2
                )
            except:
                prev_cog_position = self.cog_position
                prev_cog_orientation = self.cog_orientation
                continue
            
            if position_change < 0.01 and orientation_change < 0.01:
                i += 1
                if i > 3:    
                    break
            
            prev_cog_position = self.cog_position
            prev_cog_orientation = self.cog_orientation
            rospy.sleep(0.3)
        
    def shoot(self):
        self.shoot_msg.data = 'add'
        self.shoot_pub.publish(self.shoot_msg)
        
    def construct(self):
        if not self.construct_flag:
            rospy.loginfo("Drone: Start collecting local map")
            self.construct_flag = True
        else:
            rospy.loginfo("Drone: Collecting finished")
            self.construct_flag = False
            self.shoot_msg.data = 'save'
            self.shoot_pub.publish(self.shoot_msg)
            rospy.wait_for_message('/llm/construct_finished', Empty)
            rospy.loginfo("Drone: Received construction finished signal")
            
    def quad_planning_start(self, task):
        if not self.fix_flag:
            rospy.loginfo("Drone: Start path planning")
            self.trajectory_motion_sub = rospy.Subscriber('/llm/global_map', String, self.trajectory)
            task_= String()
            task_.data = task
            self.mission_target_pub.publish(task_)
            rospy.sleep(0.1)
            self.finish_flag = False
            self.rollback_flag = False
            self.global_map_required_pub.publish(Empty())
            rospy.wait_for_message('/llm/global_map', String)
            self.trajectory_motion_sub.unregister()
            rospy.loginfo("Drone: Path planning finished")
        else:
            rospy.loginfo("Drone: Skip drone motion")
            
    def trajectory(self, msg, retry_count=0):
        try:
            data = json.loads(msg.data)
            print(data)
            target = tool.extract_coordinates_by_type(data, 'target')
            obstacles = tool.extract_coordinates_by_type(data, 'obstacle')
            main_position,_,_ = tool.extract_coordinates_by_type(data, 'main')
            self.dog_position = np.array(main_position[0])
            current_position = ((self.cog_position.x - self.bias[0]) / self.proportion, (self.cog_position.y - self.bias[1]) / self.proportion)
            rospy.loginfo(f"Drone: Current position: {current_position}, Target position: {target}, Obstacles: {obstacles}")
            
            self.target = target
            self.obstacles = obstacles
            path = self.path_planner.generate_safe_path(current_position, target, obstacles)
            self.trajectory_motion(path)
        except Exception as e:
            max_retries = 3
            if retry_count < max_retries:
                retry_count += 1
                rospy.logwarn(f"Error in trajectory: {e}. Retrying ({retry_count}/{max_retries})...")
                rospy.sleep(1.0)  # Add delay before retry
                self.global_map_required_pub.publish(Empty())
                try:
                    new_msg = rospy.wait_for_message('/llm/global_map', String, timeout=10.0)
                    self.trajectory(new_msg, retry_count)
                except rospy.ROSException as re:
                    rospy.logerr(f"Timeout waiting for global map: {re}")
            else:
                rospy.logerr(f"Error in trajectory after {max_retries} retries: {e}. You may need to restart the simulation.")
    
    def trajectory_motion(self, path_):
        self.path = list(path_)
        last_dog_update_idx = 0      
        dog_position = self.dog_position
        i = 0

        while i < len(self.path):
            point = self.path[i]

            if self.dog_update_flag:
                self.dog_update_flag = False
                dog_position = self.dog_position
                last_dog_update_idx = i

            if self.rollback_flag:
                self.rollback_flag = False
                rospy.loginfo("Drone: Path planning stopped, task rollback")
                break
                
            if self.drone_back_flag:
                rospy.loginfo("Drone: Backtracking initiated")
                while self.drone_back_flag and i > 0:
                    i -= 1
                    self.quad_navi(np.array(self.path[i]) * self.proportion, wait=False)
                    rospy.sleep(0.3)
                rospy.loginfo("Drone: Found robot again, resuming trajectory")
                i += 1
                continue  

            move_diff = np.array(point) - np.array(self.path[last_dog_update_idx])
            predicted_dog_offset = dog_position - move_diff
            if i < len(self.path) - 1:
                future_move_diff = np.array(self.path[i + 1]) - np.array(self.path[last_dog_update_idx])
                future_dog_offset = dog_position - future_move_diff
            else:
                future_dog_offset = [0, 0]

            threshold_x = self.grid * 11 / 20
            threshold_y = self.grid * 11 / 20 * 9 / 16

            if (abs(predicted_dog_offset[0]) > threshold_x and abs(predicted_dog_offset[0]) < abs(future_dog_offset[0])) or \
                (abs(predicted_dog_offset[1]) > threshold_y and abs(predicted_dog_offset[1]) < abs(future_dog_offset[1])):
                rospy.loginfo("Drone: Potential overrun detected, waiting for dog's update")
                wait_time = 0.0
                while not self.dog_update_flag and wait_time < 10.0:
                    rospy.sleep(1)
                    wait_time += 1

                if self.dog_update_flag:
                    self.dog_update_flag = False
                    dog_position = self.dog_position
                    last_dog_update_idx = i
                    rospy.loginfo("Drone: Dog updated, resuming trajectory")
                else:
                    rospy.loginfo("Drone: No dog update within 10s, proceeding with caution")

            target_position = np.array(point) * self.proportion
            self.quad_navi(target_position, wait=False)

            rospy.sleep(0.3)
            i += 1
            
    def sample_rectangles(self, semantic_map, sampling_times):
        xmin, xmax, ymin, ymax = semantic_map
        assert (sampling_times - 1) % 4 == 0, "Sampling times must be 5, 9, 13, etc. (i.e., (sampling_times - 1) % 4 == 0)"

        n = (sampling_times - 1) // 4  # Number of rectangle layers
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        half_w, half_h = (xmax - xmin) / 2, (ymax - ymin) / 2

        points = []
        for i in range(n):
            scale = (n - i) / n  # Scale from outer to inner
            dx, dy = half_w * scale, half_h * scale
            rect = [(cx - dx, cy + dy),
                    (cx + dx, cy + dy),
                    (cx + dx, cy - dy),
                    (cx - dx, cy - dy)]
            points.extend(rect)

        points.append((cx, cy))  # Center
        return points
            
    def construct_map(self):
        if not self.fix_flag:
            self.construct()
            self.quad_takeoff()
            points = self.sample_rectangles(self.cfg['semantic_map'], self.cfg['sampling_times'])
            for point in points:
                self.quad_navi(np.array(point))
            # self.quad_navi(np.array((5,5.5)) * self.proportion)
            # self.quad_navi(np.array((-5,5.5)) * self.proportion)
            # self.quad_navi(np.array((-5,1)) * self.proportion)
            # self.quad_navi(np.array((5,1)) * self.proportion)
            # self.quad_navi((0,0))
            self.construct()
        else:
            rospy.loginfo("Drone: Skip drone motion")
        
if __name__ == "__main__":
    try:
        rospy.init_node('drone_motions')
        drone_motions = DroneMotions()
        # rospy.wait_for_message('/quadrotor/uav/cog/odom', Odometry)
        # # drone_motions.construct()
        # drone_motions.quad_takeoff()
        # drone_motions.quad_navi(np.array((5,3)) * drone_motions.proportion)
        # drone_motions.quad_navi(np.array((-5,3)) * drone_motions.proportion)
        # drone_motions.quad_navi(np.array((-5,-3)) * drone_motions.proportion)
        # drone_motions.quad_navi(np.array((5,-3)) * drone_motions.proportion)
        # drone_motions.quad_navi((0,0))
        # drone_motions.construct()
        # drone_motions.trajectory((0,0),(-6.0, 5.0), [(-3.0,0), (-5.5, -6.0)])
        points = drone_motions.sample_rectangles(drone_motions.cfg['semantic_map'], drone_motions.cfg['sampling_times'])
        # rospy.spin()
        print("Sampled points:", points)
    except rospy.ROSInterruptException & KeyboardInterrupt:
        pass
        
