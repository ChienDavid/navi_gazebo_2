#!/usr/bin/env python3
"""
Created on Wed Mar 8 10:15:04 2023
@author: Chien Van Dang

Description: main function running global path planners for Indoor navigation
"""
import math
import numpy as np

import rospy
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped

from slam import SLAM
from astar import Astar
from navi_gazebo.msg import Pathmsgs, Robotmsgs
from gridmap import OccupancyGridMap
from utils import OBSTACLE, RESOLUTION, testmapAct2Sim, convert_path_Sim2Act, angle_quaternion2euler



class GlobalPlanner:
    def __init__(self, robot, amap, method, resolution=RESOLUTION):
        # setup a map
        self.amap = amap
        self.method = method
        self.resolution = resolution

        # setup robot numbers, topics
        self.robot_no = robot
        self.another_robot = 'tb3_1' if robot == 'tb3_0' else 'tb3_0'
        self.robot_topic = '/' + str(self.robot_no) + '/'
        self.another_robot_topic = '/' + str(self.another_robot) + '/'
        self.setup_topics(self.robot_topic, self.another_robot_topic)

        # initialize robot, map
        self.last_robot = self.robot_sim
        self.last_goal = self.goal_sim
        self.aSlam = SLAM(amap=amap, resolution=self.resolution)

        rospy.loginfo("Initializing {} method".format(self.method))
        self.global_planner = Astar(amap=amap, method=self.method, resolution=self.resolution)
        rospy.loginfo("{} plans from {} to {}".format(self.method, [round(val,3) for val in self.last_robot], [round(val,3) for val in self.last_goal]))
        self.path, t0, length = self.global_planner.plan(robot_pos=self.last_robot, goal_pos=self.last_goal, newmap=amap)

    def setup_topics(self, topic, another_topic):
        self.pub_path = rospy.Publisher(topic+"globalpath", Pathmsgs, queue_size=10)
        self.pub_pathrviz = rospy.Publisher(topic+"globalpath_rviz", Path, queue_size=10)

        rospy.Subscriber(topic+"scan", LaserScan, self.callback_obstacle)
        rospy.Subscriber(topic+"robotobs", Robotmsgs, self.callback_robotobs)
        rospy.Subscriber(topic+"move_base_simple/goal", PoseStamped, self.callback_goal)
        self.check_connection(topic+"scan", LaserScan)
        self.check_connection(topic+"robotobs", Robotmsgs)
        self.check_connection(topic+"move_base_simple/goal", PoseStamped)

        if self.robot_no == 'tb3_0':
            rospy.Subscriber(another_topic+"globalpath", Pathmsgs, self.callback_anotherpath)
            self.check_connection(another_topic+"globalpath", Pathmsgs)

    def check_connection(self, topic, msg_type):
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message(topic, msg_type, timeout=1)
            except:
                rospy.loginfo("Waiting for topic {}".format(topic))

    def callback_goal(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        theta = angle_quaternion2euler(msg.pose.orientation)
        self.goal_act = np.array([x, y, theta])
        self.goal_sim = testmapAct2Sim(x, y, resolution=self.resolution)

    def callback_obstacle(self, msg):
        self.lidar_msg = msg
        self.view_range = int(msg.range_max * 20 / self.resolution)

    def callback_robotobs(self, msg):
        if self.robot_no == 'tb3_0': # robot0 tracks robot1
            self.robot_act = msg.robot0[:5]
            self.robot_sim = [int(val) for val in msg.robot0[5:]]
            robot_inf1_x = msg.obs1_inflated_X
            robot_inf1_y = msg.obs1_inflated_Y
            robot_inf2_x = msg.obs1_predicted_X
            robot_inf2_y = msg.obs1_predicted_Y
        else:                        # robot1 tracks robot0
            self.robot_act = msg.robot1[:5]
            self.robot_sim = [int(val) for val in msg.robot1[5:]]
            robot_inf1_x = msg.obs0_inflated_X
            robot_inf1_y = msg.obs0_inflated_Y
            robot_inf2_x = msg.obs0_predicted_X
            robot_inf2_y = msg.obs0_predicted_Y

        self.robot_inf1 = np.vstack((robot_inf1_x, robot_inf1_y)).T
        self.robot_inf2 = np.vstack((robot_inf2_x, robot_inf2_y)).T
        self.robot_inf1 = self.robot_inf1.tolist()
        self.robot_inf2 = self.robot_inf2.tolist()

    def callback_anotherpath(self, msg):
        self.another_x = []
        self.another_y = []
        x_act = msg.globalCoordXs
        y_act = msg.globalCoordYs
        for (xi, yi) in zip (x_act, y_act):
            x, y = testmapAct2Sim(xi, yi, resolution=self.resolution)
            self.another_x.append(x)
            self.another_y.append(y)
        #self.another_xy = np.vstack((self.another_x, self.another_y)).T

    ###################################################################################################
    def update_another_path(self, slam_map, another_x, another_y, range=10):
        robot_footprint = slam_map.inflation_obstacles(another_x[:range], another_y[:range], distance=[1, math.ceil(3/RESOLUTION)])
        slam_map.inflation_1.extend(robot_footprint)
        robot_footprint = np.array(robot_footprint)
        robot_inflation = slam_map.inflation_obstacles(robot_footprint[:,0], robot_footprint[:,1], distance=[3, math.ceil(6/RESOLUTION)])
        slam_map.inflation_2.extend(robot_inflation)
        return slam_map

    def check_goal(self, robot, goal, goal_radius=3):
        x1, y1 = robot[0], robot[1]
        x2, y2 = goal[0], goal[1]
        dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        if dist <= goal_radius:
            return True
        return False

    def arrange_path(self, pathdata, resolution=1):
        path_act = convert_path_Sim2Act(pathdata, resolution=resolution)
        # for path tracking
        path_tracking = Pathmsgs()
        path_tracking.globalCoordXs = path_act[:,0]
        path_tracking.globalCoordYs = path_act[:,1]
        
        # for RViz
        path_rviz = Path()
        path_rviz.header.stamp = rospy.Time.now()
        path_rviz.header.frame_id = 'map'
        for (x, y) in path_act:
            pose = PoseStamped()
            pose.pose.position.x = x
            pose.pose.position.y = y
            path_rviz.poses.append(pose)
        return path_tracking, path_rviz

    def execute(self):
        self.pre_path = []
        while not rospy.is_shutdown():
            # check goal
            if self.check_goal(self.robot_sim, self.goal_sim):
                rospy.loginfo("Robot has arrived the goal!")
                continue

            # check invalid start point
            if self.amap.occupancy_grid_map[self.robot_sim[0], self.robot_sim[1]] == OBSTACLE:
                self.robot_sim = self.last_robot
                
            # update new route
            if self.robot_sim != self.last_robot or self.goal_sim != self.last_goal:
                new_mission = True if self.goal_sim != self.last_goal else False
                self.last_robot, self.last_goal = self.robot_sim, self.goal_sim
                rospy.loginfo("{} plans from {} to {}".format(self.method, [round(val,3) for val in self.last_robot], [round(val,3) for val in self.last_goal]))
                # replan
                slam_map = self.aSlam.rescan(self.robot_sim, self.view_range, self.lidar_msg, self.robot_act, self.robot_inf1, self.robot_inf2, new_mission)
                if self.robot_no == 'tb3_0':
                    slam_map = self.update_another_path(slam_map, self.another_x, self.another_y)
                self.path, _, _ = self.global_planner.plan(robot_pos=self.last_robot, goal_pos=self.last_goal, newmap=slam_map)
            if len(self.path) < 2:
                self.path = self.pre_path
            else:
                self.pre_path = self.path
                
            # publish planned route
            path_tracking, path_rviz = self.arrange_path(self.path, resolution=self.resolution)
            self.pub_path.publish(path_tracking)
            self.pub_pathrviz.publish(path_rviz)
            


def main():
    # Create a ROS node
    robot = rospy.get_param('/robot_no')
    rospy.init_node(robot+"_Global_Planner")
    
    # Define a log file
    PLANNERS = ["AstarOnline", "AstarOffline"]
    PLANNER_ID = 1
    ROBOT_MODEL = "Turtlebot3"
    
    # Initialize
    rospy.loginfo("Initializing...")
    amap = OccupancyGridMap(resolution=RESOLUTION)
    global_planner = GlobalPlanner(robot, amap, PLANNERS[PLANNER_ID])

    # Execute global path planner
    rospy.loginfo("Start executing...")
    global_planner.execute()
    rospy.loginfo("Mission completed!")

if __name__ == '__main__':
    main()





