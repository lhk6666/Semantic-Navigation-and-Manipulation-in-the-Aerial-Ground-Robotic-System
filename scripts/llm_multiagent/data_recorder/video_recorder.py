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
import json
import tool
import numpy as np
import math

class VideoRecorder:
    def __init__(self):
        # Initialize attributes
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.output_file = f'/home/dragon_llm/ros/llm_ws/src/Storage_video/{self.timestamp}.mp4'
        self.output_file_grid = f'/home/dragon_llm/ros/llm_ws/src/Storage_video/{self.timestamp}_grid.mp4'
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = None
        self.out_grid = None
        self.global_msg = None
        self.local_msg = None
        self.local_update_flag = False
        height = rospy.get_param('height', 2.3)
        self.body_length = 2.2 * 2.7 / height
        
        self.bridge = CvBridge()
        self.client = GeminiPlanner()
        self.is_shutdown = False
        
    def init_ros(self):
        rospy.init_node('video_recorder')
        rospy.Subscriber('/image_rect_color/compressed', CompressedImage, self.image_callback, queue_size=1)
        rospy.Subscriber('/camera/image_raw/compressed', CompressedImage, self.image_callback, queue_size=1)
        rospy.Subscriber('/llm/global_map', String, self.global_map_callback)
        rospy.Subscriber('/llm/local_map', String, self.local_map_callback)
        rospy.Subscriber('/llm/shutdown', Empty, self.shutdown_callback)

    def shutdown_callback(self, msg):
        self.is_shutdown = True
        
    def global_map_callback(self, msg):  
        data = json.loads(msg.data)
        target = tool.extract_coordinates_by_type(data, 'target')
        obstacles = tool.extract_coordinates_by_type(data, 'obstacle')
        self.global_msg = f"Global:\nTarget Global Position: {target},\nObstacles Global Position: {obstacles}"
    
    def local_map_callback(self, msg):
        self.local_msg = msg.data
        try:
            data = json.loads(msg.data)
            orientation = tool.extract_orientation_by_parts(data)
            main_positions, carry_flag, carry_success = tool.extract_coordinates_by_type(data, 'main', 2)
            if main_positions:
                body_position = main_positions[0]
                self.local_update_flag = True
                body_position = np.array(body_position)
                manipulator_position = body_position + self.body_length * np.array([math.cos(orientation), math.sin(orientation)])
                manipulator_position = [round(manipulator_position[0], 1), round(manipulator_position[1], 1)]
                if tool.extract_coordinates_by_type(data, 'target'):
                    target_position = np.array(tool.extract_coordinates_by_type(data, 'target'))
                else:
                    target_position = np.array([0, 0])
                obstacles = tool.extract_coordinates_by_type(data, 'obstacle')
                
                tar_vector = target_position - manipulator_position
                ori_vector = manipulator_position - body_position
        
                self.local_msg = 'Local:\nBody Position - ' + str(body_position) + ',\nManipulator Position - ' + str(manipulator_position) + ',\nCurrent Orientation - ' + str(orientation) + ',\nTarget Local Position - ' + str(target_position) + ',\nObstacles Local Position - ' + str(obstacles) + ',\nDistance - ' + str(np.linalg.norm(target_position - manipulator_position)) + ',\nAlignment - ' + str(np.dot(tar_vector, ori_vector)/np.linalg.norm(ori_vector)/np.linalg.norm(tar_vector))
            rospy.sleep(3)
            self.local_update_flag = False
        except Exception as e:
            pass
    
    def add_text_to_image(self, image, text, pos=None, bottom=False, color=(0, 0, 255)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.2
        thickness = 2
        lines = text.split('\n')
        text_size = max([cv2.getTextSize(line, font, scale, thickness)[0] for line in lines], key=lambda x: x[0])
        if pos == 'right':
            pos = (image.shape[1] - text_size[0] - 10, text_size[1] + 10)
        elif pos == 'left':
            pos = (10, text_size[1] + 10)
        for i, line in enumerate(lines):
            line_size = cv2.getTextSize(line, font, scale, thickness)[0]
            if not bottom:
                line_pos = (pos[0], pos[1] + i * (line_size[1] + 10))
            else:
                line_pos = (pos[0], image.shape[0] - (len(lines) - i) * (line_size[1] + 10))
            cv2.putText(image, line, line_pos, font, scale, color, thickness, cv2.LINE_AA)
        return image
    
    def image_callback(self, msg):
        if self.is_shutdown: 
            return
    
        frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        grid_image = self.client.draw_elements(frame)
        if self.global_msg is not None:
            grid_image = self.add_text_to_image(grid_image, self.global_msg, pos='right', color=(255, 255, 255), bottom=True)
        if self.local_update_flag:
            grid_image = self.add_text_to_image(grid_image, self.local_msg, pos='left', color=(255, 255, 255))
        if self.out is None:
            height, width = frame.shape[:2]
            self.out = cv2.VideoWriter(self.output_file, self.fourcc, 5.0, (width, height))
        
        if self.out_grid is None:
            height, width = grid_image.shape[:2]
            self.out_grid = cv2.VideoWriter(self.output_file_grid, self.fourcc, 5.0, (width, height))
        
        self.out.write(frame)
        self.out_grid.write(grid_image)
        
        if self.is_shutdown:
            rospy.signal_shutdown("All tasks are finished. Stop the recorder.")
    
    def run(self):
        self.init_ros()
        print("Press '^c' to stop recording.")
        
        while not rospy.is_shutdown() and not self.is_shutdown:
            try:
                rospy.sleep(0.1)
            except Exception as e:
                break
        
        self.shutdown()
    
    def shutdown(self):
        if self.out is not None:
            self.out.release()
        if self.out_grid is not None:
            self.out_grid.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    recorder = VideoRecorder()
    recorder.run()