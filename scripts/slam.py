#!/usr/bin/env python3
"""
Created on Mon Apr  4 15:50:36 2022
@author: Chien Van Dang

Description: SLAM
"""

import math
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

from gridmap import OccupancyGridMap
from utils import OBSTACLE, RESOLUTION, heuristic, testmapAct2Sim



class SLAM:
    def __init__(self, amap: OccupancyGridMap, resolution=RESOLUTION):
        self.truth_map = amap
        self.resolution = resolution
        self.original_map = deepcopy(amap)
        self.pre_slam_map = self.original_map
        
    def lidar_scan(self, msgScan, max_lidar_distance):
        """Convert LaserScan msg to array"""
        distances = np.array([])    # [m]
        angles = np.array([])       # [rad]
        ### collect data from laserScan.msg
        for i in range(len(msgScan.ranges)):
            angle = i * msgScan.angle_increment
            if ( msgScan.ranges[i] > max_lidar_distance ):
                distance = max_lidar_distance
            elif ( msgScan.ranges[i] < msgScan.range_min ):
                distance = msgScan.range_min
                # For real robot - protection
                if msgScan.ranges[i] < 0.01:
                    distance = max_lidar_distance
            else:
                distance = msgScan.ranges[i]
            distances = np.append(distances, distance)
            angles = np.append(angles, angle)
        return (distances, angles)

    def lidar_globalObs(self, distances, angles, robot, max_lidar_distance=3.0):
        dist_sim = np.empty((0, 2))
        for (dist, ang) in zip(distances, angles):
            if dist <= max_lidar_distance:
                x_act = robot[0] + dist*np.cos(ang+robot[2])
                y_act = robot[1] + dist*np.sin(ang+robot[2])
                (x_sim, y_sim) = testmapAct2Sim(x_act, y_act, resolution=self.resolution)
                dist_sim = np.vstack((dist_sim, (x_sim, y_sim))).astype(np.int32)
        return dist_sim

    def calc_cost(self, u: (int, int), v: (int, int), slam_map):
        """Calculate the cost between two nodes"""
        if not slam_map.is_unoccupied(u) or not slam_map.is_unoccupied(v):
            return float("inf")
        return heuristic(u, v)

    def update_map(self, local_grid: Dict, slam_map):
        """Update map
        node: tuple type
        value: OBSTACLE (255) or UNOCCUPIED (0)"""
        local_obs = np.empty((0,2))
        for node, value in local_grid.items():# for each node in local map
            if value == OBSTACLE:
                if slam_map.is_unoccupied(node): # if the node changes from unoccupied to occupied
                    local_obs = np.vstack((local_obs, list(node)))
                    slam_map.set_obstacle(node)
            else:
                if not slam_map.is_unoccupied(node): # if the node changes from occupied to unoccupied
                    slam_map.remove_obstacle(node)
        return local_obs, slam_map
    
    def rescan(self, pos_sim: (int, int), view_range: int, lidar_msg, pos_act, robot_inf1, robot_inf2, new_mission=False):
        """Rescan local grid map around the global position of the robot"""
        # reset slam map
        slam_map = deepcopy(self.original_map) #if new_mission else self.pre_slam_map
        
        # get obs lethal
        (distances, angles) = self.lidar_scan(lidar_msg, 3.5)
        lidarObs_lethal = self.lidar_globalObs(distances, angles, pos_act)
        
        # get map updated
        local_observation = self.truth_map.local_observation(global_pos=pos_sim, view_range=view_range, detected_obs=lidarObs_lethal)
        local_obs, slam_map = self.update_map(local_observation, slam_map)
        
        # determine inflation, add inflation to slam map
        localObs_inflation_1 = self.truth_map.inflation_obstacles(local_obs[:,0], local_obs[:,1], distance=[1, math.ceil(3/RESOLUTION)])
        slam_map.inflation_1.extend(localObs_inflation_1)

        if len(localObs_inflation_1) > 0:
            localObs_inflation_1_array = np.array(localObs_inflation_1)
            localObs_inflation_2 = self.truth_map.inflation_obstacles(localObs_inflation_1_array[:,0], localObs_inflation_1_array[:,1], distance=[math.ceil(2/RESOLUTION), math.ceil(3/RESOLUTION)])
            slam_map.inflation_2.extend(localObs_inflation_2)
        
        # predict obstacles 
        slam_map.inflation_1.extend(robot_inf2)
        slam_map.inflation_2.extend(robot_inf1)

        #self.pre_slam_map = slam_map
        ### debug: plot slam map
        '''
        # inflation_1 = np.array(slam_map.inflation_1)
        # inflation_2 = np.array(slam_map.inflation_2)
        robot_inf1 = np.array(robot_inf1)
        robot_inf2 = np.array(robot_inf2)
        plt.cla()
        #plt.title("another_pos_sim: {}".format([round(val, 3) for val in another_pos_sim]))
        plt.plot(slam_map.obsx, slam_map.obsy, '.k')
        #plt.plot(lidarObs_lethal[:, 0], lidarObs_lethal[:, 1], '.r')
        #if len(localObs_inflation_1) > 0:
        # plt.plot(inflation_1[:, 0], inflation_1[:, 1], '.r')
        #if len(localObs_inflation_2) > 0:
        # plt.plot(inflation_2[:, 0], inflation_2[:, 1], '.g')
        plt.plot(robot_inf1[:, 0], robot_inf1[:, 1], '.y')
        plt.plot(robot_inf2[:, 0], robot_inf2[:, 1], '.b')
        plt.plot(pos_sim[0], pos_sim[1], 'db')
        plt.axis("equal")
        plt.grid()
        plt.pause(1)
        #'''
        return slam_map



