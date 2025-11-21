#!/usr/bin/env python3
import rospy
import time
from geometry_msgs.msg import Twist, PoseStamped
import math
from tf.transformations import euler_from_quaternion

class DogCalibration:
    def __init__(self):        
        rospy.init_node('go1')
        self.dog_pose = None
        rospy.Subscriber('dog/mocap/pose', PoseStamped, self.get_dog_pose, queue_size=1)
        self.pub_go1_vel = rospy.Publisher('/go1/cmd_vel', Twist, queue_size=1) 
        rospy.sleep(1)
        self.go1_twist = Twist()
        self.position_dist = -3
        self.rate = rospy.Rate(100)
        
    def get_dog_pose(self, msg):
        self.dog_pose = msg.pose
        
    def reset_dog_orientation(self, target = 0):
        # while not rospy.is_shutdown():
        angle = euler_from_quaternion([self.dog_pose.orientation.x, self.dog_pose.orientation.y, self.dog_pose.orientation.z, self.dog_pose.orientation.w])[2]
        dist_ori = target - angle
        print(dist_ori)
        if dist_ori > 2*math.pi:
            dist_ori = dist_ori - 2 * math.pi
        elif dist_ori < -2 * math.pi:
            dist_ori = dist_ori + 2 * math.pi
        self.go1_twist.angular.z = 0.6 if target-angle >= 0 else -0.6
        self.pub_go1_vel.publish(self.go1_twist)
        while abs(target - angle) > 10e-2 and not rospy.is_shutdown():
            angle = euler_from_quaternion([self.dog_pose.orientation.x, self.dog_pose.orientation.y, self.dog_pose.orientation.z, self.dog_pose.orientation.w])[2]
            # print(target - angle)
            self.rate.sleep()
            if target - angle > 2*math.pi:
                target = target - 2 * math.pi
            elif target - angle < -2 * math.pi:
                target = target + 2 * math.pi
            if abs(target - angle) < 10e-3:
                break
        self.go1_twist=Twist()
        self.pub_go1_vel.publish(self.go1_twist)
        
    def run_vel(self):
        motion_time = abs(self.position_dist/0.2)
        if self.position_dist < 0:
            self.go1_twist.linear.x = -0.2
        else:
            self.go1_twist.linear.x = 0.2
        start_x = self.dog_pose.position.x
        end_x = start_x + self.position_dist
        start_time = time.time()
        self.pub_go1_vel.publish(self.go1_twist)
        while (not rospy.is_shutdown()):
            print(self.dog_pose.position.x-end_x)
            if abs(self.dog_pose.position.x - end_x) < 10e-2:
                break
            self.rate.sleep()
        real_time = time.time()-start_time
        self.pub_go1_vel.publish(Twist())
        return real_time / motion_time
    
    def run_ang(self):
        motion_time = abs(self.position_dist/0.6)
        if self.position_dist < 0:
            self.go1_twist.angular.z = -0.6
        else:
            self.go1_twist.angular.z = 0.6
        start_ang = euler_from_quaternion([self.dog_pose.orientation.x, self.dog_pose.orientation.y, self.dog_pose.orientation.z, self.dog_pose.orientation.w])[2]
        end_ang = start_ang + self.position_dist
        print(start_ang, end_ang)
        start_time = time.time()
        self.pub_go1_vel.publish(self.go1_twist)
        while (not rospy.is_shutdown()):
            orientation = euler_from_quaternion([self.dog_pose.orientation.x, self.dog_pose.orientation.y, self.dog_pose.orientation.z, self.dog_pose.orientation.w])
            dist_ori = end_ang - orientation[2]
            print(dist_ori)
            if abs(dist_ori) < 10e-2:
                break
            self.rate.sleep()
        real_time = time.time()-start_time
        self.pub_go1_vel.publish(Twist())
        return real_time / motion_time
    
if __name__ == '__main__':
    dogc = DogCalibration()
    # Wait for the first pose message to be received
    while dogc.dog_pose is None and not rospy.is_shutdown():
        rospy.loginfo("Waiting for dog pose...")
        rospy.sleep(0.5)
    dogc.reset_dog_orientation()
    # compensate = dogc.run_vel()
    compensate = dogc.run_ang()
    print(compensate)