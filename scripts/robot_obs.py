#!/usr/bin/env python3
"""
Created on Tue Nov 21 15:19:25 2023
@author: Chien Van Dang

Description: Predict next positions of robots
"""
import math
import numpy as np
import matplotlib.pyplot as plt

import rospy
from nav_msgs.msg import Odometry

from gridmap import OccupancyGridMap
from utils import testmapAct2Sim, angle_quaternion2euler
from navi_gazebo.msg import Robotmsgs



class RobotObstacles:
    def __init__(self, amap: OccupancyGridMap, velocity, resolution=1, show_animation=False):
        # setup a map
        self.amap = amap
        self.velocity = velocity
        self.resolution = resolution
        self.show_animation = show_animation
        
        # setup robot numbers, topics
        self.robot0_msg = Robotmsgs()
        self.robot1_msg = Robotmsgs()
        self.robot0_topic = '/tb3_0/'
        self.robot1_topic = '/tb3_1/'
        self.setup_topics(self.robot0_topic, self.robot1_topic)

    def setup_topics(self, topic0, topic1):
        self.pub_robot0 = rospy.Publisher(topic0+"robotobs", Robotmsgs, queue_size=10)
        self.pub_robot1 = rospy.Publisher(topic1+"robotobs", Robotmsgs, queue_size=10)
        rospy.Subscriber(topic0+"odom", Odometry, self.callback_robot0)
        rospy.Subscriber(topic1+"odom", Odometry, self.callback_robot1)
        self.check_connection(topic0+"odom", Odometry)
        self.check_connection(topic1+"odom", Odometry)

    def check_connection(self, topic, msg_type):
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message(topic, msg_type, timeout=1)
            except:
                rospy.loginfo("Waiting for topic {}".format(topic))

    def callback_robot0(self, msg):
        x = round(msg.pose.pose.position.x, 3)
        y = round(msg.pose.pose.position.y, 3)
        theta = round(angle_quaternion2euler(msg.pose.pose.orientation), 3)
        linear = round(msg.twist.twist.linear.x, 3)
        angular = round(msg.twist.twist.angular.z, 3)
        self.robot0_act = np.array([x, y, theta, linear, angular])
        self.robot0_sim = testmapAct2Sim(x, y, resolution=self.resolution)

    def callback_robot1(self, msg):
        x = round(msg.pose.pose.position.x, 3)
        y = round(msg.pose.pose.position.y, 3)
        theta = round(angle_quaternion2euler(msg.pose.pose.orientation), 3)
        linear = round(msg.twist.twist.linear.x, 3)
        angular = round(msg.twist.twist.angular.z, 3)
        self.robot1_act = np.array([x, y, theta, linear, angular])
        self.robot1_sim = testmapAct2Sim(x, y, resolution=self.resolution)

    def motion(self, x, y, theta, vel_linear, vel_angular=0., dt=6.):
        theta = vel_angular*dt + theta
        x = dt*vel_linear*math.cos(theta) + x
        y = dt*vel_linear*math.sin(theta) + y
        return (x, y)
    
    def predict_motion(self, robot, velocity, robot_act):
        robot_new = np.empty((0, 2), dtype=int)
        vel_range = np.arange(2, velocity, 2)
        for pos in robot:
            # new_pos = self.motion(pos[0], pos[1], robot_act[2], vel)
            for vel in vel_range:
                new_pos = self.motion(pos[0], pos[1], robot_act[2], -vel, robot_act[-1])
                pos_floor = tuple([int(val) for val in new_pos])
                pos_ceil = tuple([val+1 for val in pos_floor])
                if pos_floor in robot or pos_ceil in robot:
                    continue
                robot_new = np.vstack((robot_new, pos_floor))
                robot_new = np.vstack((robot_new, pos_ceil))
        return robot_new

    def plot_env(self, robot0_inf1, robot1_inf1, robot0_inf2, robot1_inf2, robot0_new, robot1_new):
        plt.cla()
        plt.title("Robot0 {}, Robot1 {}".format([round(val,2) for val in self.robot0_sim], [round(val,2) for val in self.robot1_sim]))
        plt.plot(self.amap.obsx, self.amap.obsy, '.k')
        plt.plot(robot0_inf2[:,0], robot0_inf2[:,1], '.y', label='robot0_inf2')
        #plt.plot(robot1_inf2[:,0], robot1_inf2[:,1], '.b', label='robot1_inf2')
        plt.plot(robot0_new[:,0], robot0_new[:,1], '.r', label='robot0_new')
        #plt.plot(robot1_new[:,0], robot1_new[:,1], '.g', label='robot1_new')
        plt.plot(robot0_inf1[:,0], robot0_inf1[:,1], '.k', label='robot0_inf1')
        #plt.plot(robot1_inf1[:,0], robot1_inf1[:,1], '.k', label='robot1_inf1')
        plt.legend()
        plt.axis("equal")
        plt.grid()
        plt.pause(1)

    def execute(self):
        while not rospy.is_shutdown():
            ### get obstacles around robots
            # robot lethal (inflation 1)
            robot0_lethal = self.amap.inflation_obstacles(self.robot1_sim, [self.robot0_sim[0]], [self.robot0_sim[1]], distance=[1, 3])
            robot1_lethal = self.amap.inflation_obstacles(self.robot0_sim, [self.robot1_sim[0]], [self.robot1_sim[1]], distance=[1, 3])
            robot0_inflation_1 = np.array(robot0_lethal)
            robot1_inflation_1 = np.array(robot1_lethal)
            # robot inflation (inflation 2)
            robot0_inflation_2 = self.amap.inflation_obstacles(self.robot1_sim, robot0_inflation_1[:,0], robot0_inflation_1[:,1], distance=[2, 8])
            robot1_inflation_2 = self.amap.inflation_obstacles(self.robot0_sim, robot1_inflation_1[:,0], robot1_inflation_1[:,1], distance=[2, 8])

            ### predict obstacles coming from each robot
            robot0_new = self.predict_motion(robot0_inflation_2, self.velocity, self.robot0_act)
            robot1_new = self.predict_motion(robot1_inflation_2, self.velocity, self.robot1_act)

            ### plot
            robot0_inflation_2 = np.array(robot0_inflation_2)
            robot1_inflation_2 = np.array(robot1_inflation_2)
            if not self.show_animation:
                self.plot_env(robot0_inflation_1, robot1_inflation_1, robot0_inflation_2, robot1_inflation_2, robot0_new, robot1_new)

            ### publish
            self.robot0_msg.robot0 = [*self.robot0_act, *self.robot0_sim]
            self.robot0_msg.obs1_X = robot1_inflation_1[:,0]
            self.robot0_msg.obs1_Y = robot1_inflation_1[:,1]
            self.robot0_msg.obs1_inflated_X = robot1_inflation_2[:,0]
            self.robot0_msg.obs1_inflated_Y = robot1_inflation_2[:,1]
            self.robot0_msg.obs1_predicted_X = robot1_new[:,0]
            self.robot0_msg.obs1_predicted_Y = robot1_new[:,1]

            self.robot1_msg.robot1 = [*self.robot1_act, *self.robot1_sim]
            self.robot1_msg.obs0_X = robot0_inflation_1[:,0]
            self.robot1_msg.obs0_Y = robot0_inflation_1[:,1]
            self.robot1_msg.obs0_inflated_X = robot0_inflation_2[:,0]
            self.robot1_msg.obs0_inflated_Y = robot0_inflation_2[:,1]
            self.robot1_msg.obs0_predicted_X = robot0_new[:,0]
            self.robot1_msg.obs0_predicted_Y = robot0_new[:,1]

            self.pub_robot0.publish(self.robot0_msg)
            self.pub_robot1.publish(self.robot1_msg)



def main():
    rospy.init_node("robot_obs")
    resolution = 1.0
    velocity = 4.0
    amap = OccupancyGridMap(resolution=resolution)
    robot_obs = RobotObstacles(amap, velocity, show_animation=True)
    robot_obs.execute()



if __name__=='__main__':
    main()


