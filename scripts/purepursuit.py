#!/usr/bin/env python3
"""
Created on Wed Aug 16 15:03:57 2023
@author: Chien Dang

Description: Implementation of Pure pursuit for path tracking: differential robot
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from navi_gazebo.msg import Pathmsgs
from utils import angle_quaternion2euler, heuristic2



class PurePursuit:
    class PI:
        def __init__(self, kp=0.5, ki=0.075):
            self.kp = kp
            self.ki = ki
            self.Pterm = 0.
            self.Iterm = 0.
            self.last_error = 0.
            self.dt = 0.1          # [s] time tick
        
        def control(self, error):
            self.Pterm = error
            self.Iterm += error * self.dt
            self.last_error = error
            output = self.Pterm*self.kp + self.Iterm*self.ki
            return output

    def __init__(self, robot, g, d, speed, angle_range):
        # Pure pursuit parameters
        self.g = g
        self.d = d
        self.angle_range = angle_range

        # robot
        self.target_speed = speed # [m/s]
        self.wheel_base = 0.3     # [m] distance from rear wheel to head of the robot
        self.wheel_width = 0.287  # [m] distance between two wheels
        self.PI_v = self.PI()     # control speed by PI controller
        
        self.cmd = Twist()


        # ROS topics
        self.robot_no = robot #rospy.get_param('~robot_no')
        self.robot_topic = '/' + str(self.robot_no) + '/'
        self.setup_topics(self.robot_topic)
        
        # initialize robot
        self.last_robot = self.robot_act
        self.robot_traj = np.array([self.last_robot])
        self.pre_nearest_point_idx = None
        self.last_start = None

    def setup_topics(self, topic):
        self.cmdPub = rospy.Publisher(topic+"cmd_vel", Twist, queue_size=10)
        rospy.Subscriber(topic+"globalpath", Pathmsgs, self.callback_path)
        rospy.Subscriber(topic+"odom", Odometry, self.callback_robot)
        self.check_connection(topic+"globalpath", Pathmsgs)
        self.check_connection(topic+"odom", Odometry)
        #self.get_path("/globalpath", Pathmsgs)

    def check_connection(self, topic, msg_type):
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message(topic, msg_type, timeout=1)
            except:
                rospy.loginfo("Waiting for topic {}".format(topic))

    def callback_path(self, msg):
        traj_x = msg.globalCoordXs
        traj_y = msg.globalCoordYs
        self.traj = np.dstack((traj_x, traj_y))
        self.traj = np.squeeze(self.traj)
        self.last_idx =len(self.traj) - 1
        self.goal = [traj_x[self.last_idx], traj_y[self.last_idx]]
        self.start = [traj_x[0], traj_y[0]]

    def get_path(self, topic, msg_type):
        msg = rospy.wait_for_message(topic, msg_type, timeout=10)
        traj_x = msg.globalCoordXs
        traj_y = msg.globalCoordYs
        self.traj = np.dstack((traj_x, traj_y))
        self.traj = np.squeeze(self.traj)
        self.last_idx =len(self.traj) - 1
        self.goal = [traj_x[self.last_idx], traj_y[self.last_idx]]

    def callback_robot(self, msg):
        x = round(msg.pose.pose.position.x, 3)
        y = round(msg.pose.pose.position.y, 3)
        theta = round(angle_quaternion2euler(msg.pose.pose.orientation), 3)
        linear = round(msg.twist.twist.linear.x, 3)
        angular = round(msg.twist.twist.angular.z, 3)
        self.robot_act = [x, y, theta, linear, angular]

    def search_target_idx(self, robot):
        if self.pre_nearest_point_idx is None:
            dx = [robot[0] - x for x in self.traj[:,0]]
            dy = [robot[1] - y for y in self.traj[:,1]]
            dist = np.hypot(dx, dy)
            idx = np.argmin(dist)
            self.pre_nearest_point_idx = idx
        else:
            idx = self.pre_nearest_point_idx
            dist_to_idx = heuristic2(robot, self.traj[idx])
            while (idx + 1) < len(self.traj):
                dist_to_nextidx = heuristic2(robot, self.traj[idx+1])
                if dist_to_idx < dist_to_nextidx:
                    break
                idx = idx + 1 if (idx + 1) < len(self.traj) else idx
                dist_to_idx = dist_to_nextidx
            self.pre_nearest_point_idx = idx

        # update lookahead distance
        d = self.g*robot[3] + self.d
        #d = min(max(d, 0.2), 0.33)
        while d > heuristic2(self.robot_act, self.traj[idx]):
            if (idx + 1) >= len(self.traj):
                break
            idx += 1
        return idx, d

    def steering_control(self, robot, trajectory, pre_idx):
        # position of target
        idx, d = self.search_target_idx(robot)
        if idx < pre_idx:
            idx = pre_idx
        if idx < len(trajectory):
            [tx, ty] = trajectory[idx]
        else:
            [tx, ty] = trajectory[-1]
            idx = len(trajectory) - 1
        # angular velocity
        alpha = math.atan2(ty - robot[1], tx - robot[0]) - robot[2]
        w = math.atan2(2. * self.wheel_base * math.sin(alpha) / d, 1.)
        return w, idx

    def reached_goal(self, robot, goal_radius=0.075):
        goal = list(np.asarray(self.goal))
        if len(goal) != 2:
            print("Error goal: {}".format(self.goal))
            return False
        dist = heuristic2(robot, self.goal)
        if dist > goal_radius:
            return False
        return True

    def over_angle(self, robot, trajectory, idx, angle=math.pi/5):
        target = trajectory[idx]
        alpha = math.atan2(target[1] - robot[1], target[0] - robot[0]) - robot[2]
        delta = alpha**2 - angle**2
        if delta > 0:
            return True
        return False
    
    def move(self, linear_vel, angular_vel):
        self.cmd.linear.x = linear_vel
        self.cmd.angular.z = angular_vel
        self.cmdPub.publish(self.cmd)

    def execute(self):
        while not rospy.is_shutdown():
            # update last pose of the robot
            if self.last_robot[:3] != self.robot_act[:3]:
                self.last_robot = self.robot_act

            # check update new start point (updated course)
            if self.last_start != self.start:
                self.pre_nearest_point_idx = None
                self.target_idx, _ = self.search_target_idx(self.last_robot)
                self.last_start = self.start
            
            # check goal
            if self.reached_goal(self.last_robot) or self.target_idx >= len(self.traj):
                self.move(0., 0.)
                continue

            # calc control input
            if self.over_angle(self.last_robot, self.traj, self.target_idx, angle=self.angle_range):
                v = 0.
            else:
                vel_err = self.target_speed - self.last_robot[3]
                v = self.PI_v.control(vel_err)
            w, self.target_idx = self.steering_control(self.last_robot, self.traj, self.target_idx)
            #w = 0.9*w
            
            # plot
            self.robot_traj = np.vstack((self.robot_traj, self.last_robot))
            rospy.sleep(self.target_speed)
            #plot_scene(self.traj, self.robot_traj, self.last_robot, self.target_idx, [v, w])
            
            # move the robot
            self.move(v, w)
            
        # plot and stop
        self.move(0., 0.)
        self.robot_traj = np.vstack((self.robot_traj, self.last_robot))
        rospy.loginfo("Mission completed!")
        #rospy.loginfo("Close Figure to finish.")
        #plot_scene(self.traj, self.robot_traj, self.last_robot, self.target_idx, [0., 0.], show=True)
        

###############################################################################
def plot_scene(course, traj, robot, idx, vel, show=False):
    plt.cla()
    plt.title("Path tracking controller")
    plt.plot(course[:,0], course[:,1], "-r", label="course")
    plt.plot(traj[:,0], traj[:,1], "-g", label="trajectory")
    if idx < len(course):
        plt.plot(course[idx][0], course[idx][1], "xr", label="target")
    plt.plot(robot[0], robot[1], "dg", label="robot")
    plot_arrow(*robot[:3])
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    if not show:
        plt.pause(0.001)
    else:
        plt.show()

def plot_arrow(x, y, yaw, length=0.1, width=0.1, fc="g", ec="k"):
    if not isinstance(x, float):
        for ix, iy, iyaw in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)


###############################################################################
def main():
    # Create a ROS node
    robot = rospy.get_param('/robot_no')
    rospy.init_node(robot+"_Path_Tracker")
    
    # Initialize
    rospy.loginfo("Initializing...")
    g = 0.2               # look forward gain
    d = 0.3               # [m] lookahead distance
    target_speed = 0.10   # [m/s] speed of robot
    angle_range = math.pi/4
    pp_controller = PurePursuit(robot, g, d, target_speed, angle_range=angle_range)

    # Execute path tracker
    rospy.loginfo("Start executing...")
    pp_controller.execute()
    rospy.loginfo("Mission completed!")



if __name__=="__main__":
    main()





