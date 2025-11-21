#!/usr/bin/env python3
import sys
import rospy
import tool
from std_msgs.msg import Empty, Bool
from motions import DogMotions, DroneMotions, DogMotionsSimulated
from reasoners import GeminiReasoner
import threading

class gpt_main:
    def __init__(self):
        rospy.init_node('gpt_main')
        resolution = rospy.get_param('resolution', 1080)
        arg_fix_posi = rospy.get_param('fix_posi_param', 'drone')
        height = rospy.get_param('height', 2.3)
        FOV = rospy.get_param('FOV', 1.57)
        bias_x = rospy.get_param('bias_x', 1.0)
        bias_y = rospy.get_param('bias_y', 1.0)
        bias = [bias_x, bias_y]
        simulation = rospy.get_param('simulation', False)
        map_updater = rospy.get_param('map_updater', False)

        self.image_update_flag = False
        self.image_start_flag = False
        self.rollback_flag = False
        self.map_updater_flag = map_updater
        self.grasp_state = 'Not yet catch the object'
        self.map_updater_pub = rospy.Publisher('/llm/map_updater', Bool, queue_size=1)
        self.shutdown_pub = rospy.Publisher('/llm/shutdown', Empty, queue_size=1)

        #Get parameter  
        self.drone_motions = DroneMotions(resolution=resolution, arg_fix_posi=arg_fix_posi, height=height, FOV=FOV, bias=bias)
        if simulation == False:
            self.dog_motions = DogMotions(resolution=resolution, height=height, FOV=FOV, bias=bias)
        else:
            self.dog_motions = DogMotionsSimulated(resolution=resolution, height=height, FOV=FOV, bias=bias)

        #Gemini
        self.client = GeminiReasoner()

        #Thread
        self.rollback_sub = rospy.Subscriber('/llm/rollback', Empty, self.rollback)
        self.input()
        
    def rollback(self, msg):
        self.rollback_flag = True
        rospy.logerr('Global planner: Carrying Failure - Rollback to attach the object again')

    def input(self):  
        try:
            try:
                rospy.loginfo("Global Planner: Please input your command.")
                message_t = input()
            except KeyboardInterrupt:
                rospy.signal_shutdown("Global Planner: User requested shutdown")
                sys.exit()
            if 'exit' in message_t or 'q' in message_t:
                rospy.signal_shutdown("Global Planner: User requested shutdown")
                sys.exit()         
            decision_text = self.client.global_planner(message_t)
            rospy.loginfo(decision_text)
            decision_list = tool.convert_string_to_list(decision_text)
            if len(decision_list) > 5:
                self.map_updater_flag = True
            self.map_updater_pub.publish(self.map_updater_flag)
            #Sub-task execution
            i = 0
            while (not rospy.is_shutdown()):
                try:
                    if self.rollback_flag == False:
                        command = decision_list[i]
                    else:
                        i -= 3
                        command = decision_list[i]
                        self.rollback_flag = False
                    i += 1                   
                    rospy.loginfo(f"Global Planner: Processing {command}")
                except Exception:
                    rospy.loginfo("Global Planner: All sub-task have been processed. Exiting the while loop.")
                    self.shutdown_pub.publish(Empty())
                    break
                mf = self.client.mf_planner(command)
                rospy.loginfo(f"Global Planner: Executing {mf}")
                try:
                    exec(mf.strip('```'))
                except Exception as e:
                    rospy.loginfo(f'Global Planner: Error in exec, retrying.... Reason: {e}')
                    i -= 1
        except KeyboardInterrupt:
            rospy.signal_shutdown("Global Planner: User requested shutdown")


if __name__=='__main__':
    node = gpt_main()
    rospy.spin()