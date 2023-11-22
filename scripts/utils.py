#!/usr/bin/env python3
"""
Created on Fri Apr  1 14:55:50 2022
@author: Chien Van Dang

Description: Utility functions
"""

import math
import numpy as np
from typing import List

OBSTACLE = 255
UNOCCUPIED = 0



class Path():
    """A path stores its cost and trajectory of robot: xlist, ylist, directionlist, cost"""
    def __init__(self, xlist, ylist, cost):
        self.xlist = xlist
        self.ylist = ylist
        self.cost = cost

class Node:
    def __init__(self, x, y, cost, parent_idx):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_idx = parent_idx
        
    def __str__(self):
        return str(self.x) + ", " + str(self.y) + ", " + str(self.cost) + ", " + str(self.parent_idx)

def heuristic(p: (int, int), q: (int, int)) -> float:
    """Return distance between two nodes: Euclidean distance"""
    x1, y1 = p[0], p[1]
    x2, y2 = q[0], q[1]
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def heuristic2(p, q) -> float:
    """Return distance between two poses: Euclidean distance"""
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return math.hypot(dx, dy)

def get_movements_4n(x, y, r=1) -> List:
    """Return neighbors' positions from a vertex (x,y): 4 traversable vertices"""
    return [[x+r, y+0],
            [x+0, y+r],
            [x-r, y+0],
            [x+0, y-r]]

def get_movements_8n(x, y, r=1) -> List:
    """Return neighbors' positions from a vertex (x,y): 8 traversable vertices"""
    return [(x+r, y+0),
            (x+r, y+r),
            (x+0, y+r),
            (x-r, y+r),
            (x-r, y+0),
            (x-r, y-r),
            (x+0, y-r),
            (x+r, y-r)]

def check_goal(p: (float, float), q: (int, int)):
    if abs(q[0] - p[0]) < 1.0 and abs(q[1] - p[1]) < 1.0:
        return True
    return False



def testmapAct2Sim(x: float, y: float, resolution=1):
    res_x = int((95 - 20*x)/resolution)
    res_y = int((55 - 20*y)/resolution)
    return (res_x, res_y)

def testmapSim2Act(x: int, y: int, resolution=1):
    res_x = float((95 - x * resolution) / 20)
    res_y = float((55 - y * resolution) / 20)
    return (round(res_x, 5), round(res_y, 5))

def convert_path_Sim2Act(path, resolution=1):
    res = np.empty((0, 2))
    for coord in path:
        x, y = testmapSim2Act(coord[0], coord[1], resolution=resolution)
        res = np.vstack((res, (float(x), float(y))))
    return res

def angle_quaternion2euler(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    w = math.atan2(siny_cosp, cosy_cosp)
    yaw = w * 180/math.pi
    return w #yaw

def convert_global2local(global_pose: tuple, robot_pose: tuple):
    """Convert a pose from global coordinate to robot local coordinate"""
    delta = math.atan2(global_pose[1] - robot_pose[1], global_pose[0] - robot_pose[0])
    theta_local = delta - robot_pose[2]
    d = math.sqrt((global_pose[0] - robot_pose[0])**2 + (global_pose[1] - robot_pose[1])**2)
    x_local = d * math.cos(theta_local)
    y_local = d * math.sin(theta_local)
    #theta_local = theta_local * 180 / math.pi
    return (x_local, y_local, theta_local)

def get_nearby_point(point, neighbors):
    array = np.asarray(neighbors)
    diff = np.abs(array - point)
    dist = np.hypot(diff[:,0], diff[:,1])
    idx = dist.argmax()
    return neighbors[idx]



