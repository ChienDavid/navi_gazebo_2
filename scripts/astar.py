#!/usr/bin/env python3
"""
Created on Mon May 15 11:15:06 2023
@author: Chien Van Dang

Description: Implementation of Astar (online/offline) path planning algorithm for Indoor navigation
"""

import time
import heapq
import matplotlib.pyplot as plt

import rospy

from utils import Node, heuristic
from gridmap import OccupancyGridMap
from dynamicprogramming import DP


show_animation = True
ROBOT_RADIUS = 1.0

###################################################################################################

class Astar:
    def __init__(self, amap: OccupancyGridMap, method="AstarOffline", resolution=1):
        # initialize a grid map
        self.resolution = resolution
        self.method = method
        self.sensed_map = amap
        self.configMap = self.sensed_map.config
        self.inflation_1 = self.sensed_map.inflation_1
        self.inflation_2 = self.sensed_map.inflation_2

        # initialize start, goal
        self.s_start = (0, 0)
        self.s_goal = (150, 100)
        self.startNode = Node(self.s_start[0], self.s_start[1], 0, -1)
        self.goalNode = Node(self.s_goal[0], self.s_goal[1], 0, -1)
        
        # offline method
        if self.method=="AstarOffline":
            self.dp = DP(self.sensed_map, resolution=self.resolution, show_animation=False)
            radius = int(heuristic(self.s_goal, self.s_start) * 1.5)
            self.hcost_dp = self.dp.calc_heuristic_cost(self.s_goal, self.sensed_map, radius=radius)

        self.startIdx = self.calc_idx(self.startNode, self.configMap)
        self.startCost = self.calc_total_cost(self.startNode, self.goalNode)
        
    def calc_node(self, node, pose, configMap):
        childX, childY = pose[0], pose[1]
        childCost = node.cost + self.calc_heuristic((node.x, node.y), pose)
        parentIdx = self.calc_idx(node, configMap)
        nextNode = Node(childX, childY, childCost, parentIdx)
        return nextNode

    def calc_idx(self, node, configMap):
        idx = (node.y - configMap.y_min)*configMap.x_w + (node.x - configMap.x_min)
        if idx <= 0:
            print("Error idx of the node:", idx)
        return idx
    
    def calc_heuristic(self, u: (int, int), v: (int, int)):
        if not self.sensed_map.is_unoccupied(u) or not self.sensed_map.is_unoccupied(v):
            return float("inf")
        return heuristic(u, v)
    
    def calc_total_cost(self, node, goalNode):
        """Calculate total cost function: f = g + h"""
        inflation_cost = 0.
        if (node.x, node.y) in self.inflation_2:
            inflation_cost = 200.
        elif (node.x, node.y) in self.inflation_1:
            inflation_cost = float('inf')
        if self.method == "AstarOffline":
            idx = self.calc_idx(node, self.configMap)
            hcost = self.hcost_dp[idx].cost if idx in self.hcost_dp else float("inf")
        else:
            hcost = self.calc_heuristic((node.x, node.y), (goalNode.x, goalNode.y))
        fcost = node.cost + hcost + inflation_cost
        #print("=> g, h, inf costs: {}".format([node.cost, hcost, inflation_cost]))
        return fcost
    
    def final_path(self, goal, closedDict):
        """Backtracking the path from goal to start"""
        path = []
        nodeIdx = goal.parent_idx
        while nodeIdx != -1:
            node = closedDict[int(nodeIdx)]
            path.append((node.x, node.y))
            nodeIdx = node.parent_idx
        path = list(reversed(path))
        if len(path) <= 1:
            path.append((goal.x, goal.y))
        return path

    def plan(self, robot_pos: (int, int), goal_pos: (int, int), newmap: OccupancyGridMap):
        time_start = time.time()
        # update map
        self.sensed_map = newmap
        self.inflation_1 = self.sensed_map.inflation_1
        self.inflation_2 = self.sensed_map.inflation_2
        
        if self.method=="AstarOffline" and self.s_goal != goal_pos:
            radius = int(heuristic(goal_pos, robot_pos) * 1.5)
            self.hcost_dp = self.dp.calc_heuristic_cost(goal_pos, self.sensed_map, radius=radius)
        
        # update start, goal
        if self.s_goal != goal_pos or self.s_start != robot_pos:
            if self.s_goal != goal_pos:
                self.s_goal = goal_pos
                self.goalNode = Node(self.s_goal[0], self.s_goal[1], 0, -1)
            if self.s_start != robot_pos:
                self.s_start = robot_pos
                self.startNode = Node(self.s_start[0], self.s_start[1], 0, -1)
                self.startIdx = self.calc_idx(self.startNode, self.configMap)
            self.startCost = self.calc_total_cost(self.startNode, self.goalNode)
        
        # update queue
        self.openDict, self.closedDict = {}, {}
        self.openDict[self.startIdx] = self.startNode
        self.priority_queue = []
        heapq.heappush(self.priority_queue, (self.startCost, self.startIdx))
        
        while True:
            if len(self.priority_queue) > 0:
                currentCost, currentIdx = heapq.heappop(self.priority_queue)
            else:
                print("Please identify valid start/goal points!")
                break
            if currentIdx in self.openDict:
                currentNode = self.openDict.pop(currentIdx)
                self.closedDict[currentIdx] = currentNode
            else:
                continue
            
            # check goal
            if currentNode.x == self.goalNode.x and currentNode.y == self.goalNode.y:
                self.goalNode = currentNode
                break
            
            successors = self.sensed_map.get_neighbors((currentNode.x, currentNode.y), avoid_obstacles=True)
            for pose in successors:
                node = self.calc_node(currentNode, pose, self.configMap)
                nodeIdx = self.calc_idx(node, self.configMap)
                if nodeIdx in self.closedDict:
                    continue
                if nodeIdx not in self.openDict or self.openDict[nodeIdx].cost > node.cost:
                    self.openDict[nodeIdx] = node
                    nodeCost = self.calc_total_cost(node, self.goalNode)
                    heapq.heappush(self.priority_queue, (nodeCost, nodeIdx))
                    
                    # plot
                    if not show_animation and len(self.closedDict) % 100 == 0:
                        plt.cla()
                        plt.title("A*: expanded nodes={}, currentNode={}".format(len(self.closedDict), [currentNode.x, currentNode.y]))
                        plt.plot(self.sensed_map.obsx, self.sensed_map.obsy, '.k')
                        for idx, node in self.openDict.items():
                            plt.plot(node.x, node.y, '.g')
                        for idx, node in self.closedDict.items():
                            plt.plot(node.x, node.y, '.b')
                        plt.grid(True)
                        plt.axis("equal")
                        plt.pause(0.0001)
        # backtracking the path
        path = self.final_path(self.goalNode, self.closedDict)
        time_stop = time.time()
        t = round(time_stop - time_start, 3)
        return path, t, len(self.closedDict)



###################################################################################################
def main():
    print("Implementation of A-star Algorithm!...")
    rospy.init_node("Astar")
    
    amap = OccupancyGridMap()
    astar = Astar(amap)
    start = (135, 55)
    goal = (30, 150)
    print("Plan a path...")
    apath, executedTime, closedDict = astar.plan(start, goal, amap)
    
    # plot
    print("Plot...")
    if show_animation:
        plt.cla()
        #plt.title("A* Planner: length={}m, Exec. time={}s".format(round(apath.cost, 3), executedTime))
        plt.plot(astar.obsx, astar.obsy, 'sk')
        #plt.plot(apath.xlist, apath.ylist, '-g')
        for i in range(len(apath)):
            plt.plot(apath[i][0], apath[i][1], '.g')
        plt.plot(start[0], start[1], 'dr')
        plt.plot(goal[0], goal[1], 'xr')
        plt.grid(True)
        plt.axis("equal")
        plt.show()
    print("Finished!")

if __name__ == '__main__':
    main()



