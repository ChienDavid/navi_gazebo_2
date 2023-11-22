#!/usr/bin/env python3
"""
Created on Wed Jun 9 10:57:31 2021
@author: Chien Van Dang

Description: Implementation of Dynamic Programming for computing heuristic costs
"""

import csv
import math
import heapq
import numpy as np
import matplotlib.pyplot as plt

import rospy

from gridmap import OccupancyGridMap
from utils import heuristic


OBSTACLE = 255
UNOCCUPIED = 0

###############################################################################
class Node:
    def __init__(self, x, y, yaw, cost, parent_idx):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.cost = cost
        self.parent_idx = parent_idx
        
    def __str__(self):
        return str(self.x) + ", " + str(self.y) + ", " + str(self.yaw) + ", " + str(self.cost) + ", " + str(self.parent_idx)

def move_cost():
    """Action move and its cost: dx, dy, cost"""
    move = [[1, 0, 1],
            [0, 1, 1],
            [-1, 0, 1],
            [0, -1, 1],
            [-1, -1, math.sqrt(2)],
            [-1, 1, math.sqrt(2)],
            [1, -1, math.sqrt(2)],
            [1, 1, math.sqrt(2)]]
    return move

def calc_idx(node, config):
    return (node.y - config.y_min)*config.x_w + (node.x - config.x_min)

def get_theta(x1, y1, x2, y2):
    """return angle of a line defined by point1(x1, y1) to point2(x2, y2)"""
    return math.atan2(y2-y1, x2-x1)
    
def verify_node(node, gridMap, config):
    """Verify a node if it is in obstacle map and if it is at obstacle position"""
    if node.x < config.x_min or node.y < config.y_min or node.x >= config.x_max or node.y >= config.y_max:
        return False
    if gridMap[node.x, node.y] == OBSTACLE:
        return False
    return True

###############################################################################
class DP:
    def __init__(self, amap: OccupancyGridMap, resolution=1, show_animation=False):
        self.amap = amap
        self.resolution = resolution
        self.gridMap = amap.occupancy_grid_map
        self.configMap = amap.config
        height, width = amap.height, amap.width
        self.dp_map = np.zeros((int(height/self.resolution), int(width/self.resolution)), dtype=np.float32)

        self.show_animation = show_animation
        self.move = move_cost()


    def calc_heuristic_cost(self, new_goal, sensed_map, radius=300.):
        self.goal = new_goal
        self.goalNode = Node(self.goal[0], self.goal[1], 0., 0, -1)
        self.gridMap = sensed_map.occupancy_grid_map

        open_dict, closed_dict = {}, {}
        goal_idx = calc_idx(self.goalNode, self.configMap)
        open_dict[goal_idx] = self.goalNode
        priority_queue = [(0, goal_idx)]

        done = False
        while not done:
            if not priority_queue:
                break

            # select a node
            current_cost, current_idx = heapq.heappop(priority_queue)
            if current_idx in open_dict:
                current_node = open_dict.pop(current_idx)
                closed_dict[current_idx] = current_node
            else:
                continue

            if heuristic(self.goal, (current_node.x, current_node.y)) > radius:
                continue
                    
            # expand node based on movement
            for i in range(len(self.move)):
                x = current_node.x + self.move[i][0]
                y = current_node.y + self.move[i][1]
                theta = get_theta(x, y, current_node.x, current_node.y)
                
                cost = current_node.cost + self.move[i][2]
                child_node = Node(x, y, theta, cost, current_idx)
                child_idx = calc_idx(child_node, self.configMap)
                
                if child_idx in closed_dict:
                    continue
                if not verify_node(child_node, self.gridMap, self.configMap):
                    continue
                
                if child_idx not in open_dict or open_dict[child_idx].cost >= child_node.cost:
                    open_dict[child_idx] = child_node
                    heapq.heappush(priority_queue, (child_node.cost, child_idx))
                    self.dp_map[y, x] = child_node.cost

        # plot: show heatmap in grayscale
        if self.show_animation:
            fig, ax = plt.subplots()
            ax.cla()
            original_cmap = plt.cm.get_cmap('gray')
            reversed_cmap = original_cmap.reversed()
            im = ax.pcolormesh(self.dp_map, cmap=reversed_cmap)
            ax.set_title("DP field: goal {}".format(self.goal))
            fig.colorbar(im)
            plt.show()
    
        return closed_dict


###############################################################################
def main():
    print("Implementation of Dynamic programming Algorithm!")
    rospy.init_node("DP")
    
    rospy.loginfo("Initialization...")
    amap = OccupancyGridMap()
    dp = DP(amap, show_animation=True)

    rospy.loginfo("Calc. heuristic cost...")
    goal = [30, 150, np.deg2rad(90)]
    dp_cost = dp.calc_heuristic_cost(goal, amap)

    # write dp cost to csv file
    rospy.loginfo("Start writing to file!...")
    with open("dp_cost.csv", 'w', newline='') as afile:
        writer = csv.writer(afile)
        writer.writerow(["x", "y", "cost"])
        for idx, val in dp_cost.items():
            writer.writerow([val.x, val.y, val.cost])
    rospy.loginfo("Finish writing to file!")
    rospy.loginfo("Finished!")



if __name__ == '__main__':
    main()




