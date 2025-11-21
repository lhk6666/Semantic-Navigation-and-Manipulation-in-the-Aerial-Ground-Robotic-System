#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from __future__ import print_function 
import sys, select, termios, tty
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from motions.dog_motion import DogMotions

class KeyboardControl:
    def __init__(self):
           self.settings = termios.tcgetattr(sys.stdin)
           rospy.init_node("keyboard_control")
           self.dog_motions = DogMotions()

           self.nav_pub = rospy.Publisher('/go1/cmd_vel', Twist, queue_size=1)
           self.stand_pub = rospy.Publisher('/go1/stand', Empty, queue_size=1)
           self.sit_pub = rospy.Publisher('/go1/sit', Empty, queue_size=1)

           self.xy_vel = rospy.get_param("xy_vel", 0.2)
           self.yaw_vel = rospy.get_param("yaw_vel", 0.4)

           self.nav_msg = Twist()
           self.msg = """
Instruction:

---------------------------

     q           w           e           u         i
(turn left)  (forward)  (turn right)  (attach)  (detach)

     a           s           d          j       k       l
(move left)  (backward) (move right)  (stop) (stand)  (sit)


Please don't have caps lock on.
CTRL+c to quit
---------------------------
"""

    def getKey(self):
           tty.setraw(sys.stdin.fileno())
           select.select([sys.stdin], [], [], 0)
           key = sys.stdin.read(1)
           termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
           return key

    def printMsg(self, msg, msg_len=50):
           print(msg.ljust(msg_len) + "\r", end="")

    def run(self):
           print(self.msg)
           try:
                  while True:
                         key = self.getKey()
                         msg = ""

                         if key == 'w':
                                self.nav_msg.linear.x += self.xy_vel
                                self.nav_pub.publish(self.nav_msg)
                                msg = "send +x vel command"
                         if key == 's':
                                self.nav_msg.linear.x += -self.xy_vel
                                self.nav_pub.publish(self.nav_msg)
                                msg = "send -x vel command"
                         if key == 'a':
                                self.nav_msg.linear.y += self.xy_vel
                                self.nav_pub.publish(self.nav_msg)
                                msg = "send +y vel command"
                         if key == 'd':
                                self.nav_msg.linear.y += -self.xy_vel
                                self.nav_pub.publish(self.nav_msg)
                                msg = "send -y vel command"
                         if key == 'q':
                                self.nav_msg.angular.z += self.yaw_vel
                                self.nav_pub.publish(self.nav_msg)
                                msg = "send +yaw vel command"
                         if key == 'e':
                                self.nav_msg.angular.z += -self.yaw_vel
                                self.nav_pub.publish(self.nav_msg)
                                msg = "send -yaw vel command"
                         if key == 'j':
                                self.nav_msg.linear.x = 0
                                self.nav_msg.linear.y = 0
                                self.nav_msg.angular.z = 0
                                self.nav_pub.publish(self.nav_msg)
                                msg = "stop!"
                         if key == 'k':
                                self.stand_pub.publish(Empty())
                                msg = "stand!"
                         if key == 'l':
                                self.sit_pub.publish(Empty())
                                msg = "sit!"
                         if key == 'u':
                                self.dog_motions.catch_objects()
                         if key == 'i':
                                self.dog_motions.detach_objects()

                         if key == '\x03':
                                break

                         self.printMsg(msg)
                         rospy.sleep(0.001)

           except Exception as e:
                  print(repr(e))
           finally:
                  termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

if __name__ == "__main__":
    kc = KeyboardControl()
    kc.run()
