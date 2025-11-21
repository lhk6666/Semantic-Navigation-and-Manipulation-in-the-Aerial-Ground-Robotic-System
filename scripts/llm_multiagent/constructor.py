#!/usr/bin/env python3
import cv2
from cv_bridge import CvBridge
import rospy
from std_msgs.msg import String, Empty, Bool
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
import math
import time
import base64
import threading
import concurrent.futures
from planners import GeminiPlanner
from reasoners import GeminiReasoner

class WorldConstructor:
    def __init__(self):
        resolution = rospy.get_param('resolution', 1080)
        height = rospy.get_param('height', 2.3)
        FOV = rospy.get_param('FOV', 1.57)
        bias_x = rospy.get_param('bias_x', 1.0)
        bias_y = rospy.get_param('bias_y', 1.0)
        self.bias = [bias_x, bias_y]

        self.bridge = CvBridge()
        self.gemini_reasoner = GeminiReasoner()
        self.gemini_planner = GeminiPlanner()
        self.threading_pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)

        rospy.Subscriber('/quadrotor/uav/cog/odom', Odometry, self.cog_register, queue_size=1)
        rospy.Subscriber('/image_rect_color/compressed', CompressedImage, self.image_receiver, queue_size=1)
        rospy.Subscriber('/camera/image_raw/compressed', CompressedImage, self.image_receiver, queue_size=1)
        rospy.Subscriber('/llm/map_updater', Bool, self.map_updater_flag_cb)
        rospy.Subscriber('/llm/mission_target', String, self.mission_target)
        rospy.Subscriber('/llm/local_map_required', Empty, self.local_map_required)
        rospy.Subscriber('/llm/global_map_required', Empty, self.global_map_required)
        rospy.Subscriber('/llm/shoot', String, self.shoot)
        rospy.Subscriber('/llm/shutdown', Empty, self.shutdown_callback)
        
        self.map_pub = rospy.Publisher('/llm/global_map', String, queue_size=1)
        self.local_map_pub = rospy.Publisher('/llm/local_map', String, queue_size=1)
        self.construct_pub = rospy.Publisher('/llm/construct_finished', Empty, queue_size=1)
        
        self.component = ''
        hfgrid = 12 * resolution / 1080
        self.proportion = height/hfgrid * math.tan(FOV/2)
        self.update_timegap = 10
        
        rospy.loginfo("Ready!!!")
        
    def global_map_required(self, msg):
        global_map_msg = String()
        global_map_text = self.gemini_reasoner.map_getter()
        task_spc_global_map = self.gemini_planner.classifier(task=self.mission_target_text, data=global_map_text)
        global_map_msg.data = task_spc_global_map
        self.map_pub.publish(task_spc_global_map)
        
    def local_map_required(self, msg):
        rospy.loginfo('Local Constructor: Start')
        start = time.time()
        cv2_image_ = self.cv_image
        cv2_image = self.gemini_planner.draw_elements(cv2_image_)
        cv2_image = cv2.resize(cv2_image, (1920, 1080))
        image_base64 = base64.b64encode(cv2.imencode('.jpeg', cv2_image)[1]).decode()
        response = self.gemini_planner.classifier_fast(self.mission_target_text,image_base64)
        local_map_msg = String()
        local_map_msg.data = response
        rospy.sleep(0.2)
        rospy.loginfo('Local Constructor: Finish. --Time consumption:' + str(time.time()-start))
        self.local_map_pub.publish(local_map_msg)
    
    def shoot(self,msg):
        if not msg.data == 'save':
            rospy.loginfo("Global Constructor: Received local map")
            cog_position = self.cog_position
            position_text = '(' + str(round((cog_position.x-self.bias[0])/self.proportion, 1)) + ', ' + str(round((cog_position.y-self.bias[1])/self.proportion, 1)) + ')'
            cv_image_ = self.cv_image
            cv_image = self.gemini_planner.draw_elements(cv_image_)
            cv_image = cv2.resize(cv_image, (1920, 1080))
            image_base64 = base64.b64encode(cv2.imencode('.jpeg', cv_image)[1]).decode()
            self.threading_pool.submit(self.assist, image_base64, position_text)
        
        elif msg.data == 'save':
            self.threading_pool.shutdown(wait=True)
            rospy.loginfo("Global Constructor: All Local maps are received. Processing constructing")
            start_time = time.time()
            global_map_text = self.gemini_reasoner.map_constructor(self.component)
            rospy.loginfo('Global Constructor: \nAccording to\n' + self.component + '\n' + global_map_text + '\n' + 'Time Consumption: ' + str(time.time()-start_time))
            self.construct_pub.publish(Empty())
            self.component = ''
            self.map_update_thread = threading.Thread(target=self.map_updater)
            self.map_update_thread.start()
            self.map_updating_flag = False
                      
    def assist(self, image_base64, msg_data):
        start_time = time.time() 
        _, nonjson_response = self.gemini_planner.describer(image_base64, json_format=False)
        temp = f'When the world coordinates of the central of image is: {msg_data}.\nThe local coordinate of objects in the image is:\nf{nonjson_response}\n'
        self.component += temp
        rospy.loginfo(f'Global Constructor: Global coordinate {msg_data} finished. --Time consumption: {time.time()-start_time}')

    def map_updater_flag_cb(self, msg):
        self.map_updater_flag = msg.data
        if self.map_updater_flag:
            rospy.loginfo(f'Global Constructor: World map will be updated for every {self.update_timegap} seconds')
        else:
            rospy.loginfo('Global Constructor: World map will not be updated for simple task')
        
    def map_updater(self):
        global_map_text = self.gemini_reasoner.map_getter()  
        while not rospy.is_shutdown() and self.map_updater_flag:
            rospy.sleep(self.update_timegap)
            cv2_image_ = self.cv_image
            cog_position = self.cog_position
            cv2_image = self.gemini_planner.draw_elements(cv2_image_)
            cv2_image = cv2.resize(cv2_image, (1920, 1080))
            image_base64 = base64.b64encode(cv2.imencode('.jpeg', cv2_image)[1]).decode()    
            try:
                response, nonjson_response = self.gemini_planner.describer(image_base64, json_format=False)
                cord = str(round((cog_position.x-self.bias[0]) / self.proportion, 1)) + ', ' + str(round((cog_position.y - self.bias[1]) / self.proportion, 1))
                global_map_text, local_info = self.gemini_reasoner.map_updater(nonjson_response, cord)
                rospy.loginfo('Global Constructor:' + '\nUpdate global map:\n' + global_map_text + '\n')
            except:
                rospy.loginfo('Global Constructor: No update')
                continue

    def image_receiver(self, msg):
        self.cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        cv2.imshow('image', self.cv_image)
        cv2.waitKey(1)

    def mission_target(self, msg):
        self.mission_target_text = msg.data
            
    def cog_register(self, msg):
        self.cog_position = msg.pose.pose.position

    def shutdown_callback(self, msg):
        self.map_updater_flag = False
        rospy.loginfo('Global Constructor: Shutdown. Because all tasks are finished')          

if __name__ == '__main__':
    try:
        rospy.init_node('image_constructor', anonymous=True)
        image_constructor = WorldConstructor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass