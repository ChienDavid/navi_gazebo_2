#!/usr/bin/env python3
"""
Created on Fri Apr  1 15:42:36 2022
@author: Chien Van Dang

Description: Occupancy Grid Map
"""

import rospy
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from nav_msgs.msg import OccupancyGrid

from utils import get_movements_4n, get_movements_8n, testmapSim2Act, heuristic


OBSTACLE = 255
UNOCCUPIED = 0

class OccupancyGridMap:
    class Config:
        def __init__(self, obsx, obsy):
            self.x_max = max(obsx)
            self.y_max = max(obsy)
            self.x_min = min(obsx)
            self.y_min = min(obsy)
            self.x_w = round(self.x_max - self.x_min)
            self.y_w = round(self.y_max - self.y_min)

    def __init__(self, resolution=1, exploration_setting='8N'):
        """Set initial values for the map occupancy grid"""
        self.resolution = resolution
        self.exploration_setting = exploration_setting
        rospy.Subscriber("/map", OccupancyGrid, self.callback_map)
        self.check_connection("/map", OccupancyGrid)

        msg = rospy.wait_for_message("/map", OccupancyGrid, timeout=1)
        self.height = msg.info.height
        self.width = msg.info.width
        self.occupancy_grid_map = np.zeros((int(self.width/self.resolution), int(self.height/self.resolution)), dtype=np.float32)

        self.obsx = []
        self.obsy = []
        self.obs = np.empty((0, 2))
        self.execute(self.height, self.width)

    def check_connection(self, topic, msg_type):
        msg = None
        while msg is None:
            try:
                msg = rospy.wait_for_message(topic, msg_type, timeout=1)
            except:
                rospy.loginfo("Topic {} not ready yet, retrying for setting up".format(topic))

    def callback_map(self, msg):
        self.data = msg.data
        
    def execute(self, height, width):
        # lethal layer
        for h in range(height):
            for w in range(width):
                i = h * width + w
                if self.data[i] == 100:
                    x_sim = int((width-w)/self.resolution)
                    y_sim = int((height-h)/self.resolution)
                    self.occupancy_grid_map[x_sim, y_sim] = OBSTACLE
                    self.obsx.append(x_sim)
                    self.obsy.append(y_sim)
                    x_act, y_act = testmapSim2Act(width-w, height-h)
                    self.obs = np.vstack((self.obs, (x_act, y_act)))
        self.config = self.Config(self.obsx, self.obsy)
        # inflation layer
        self.inflation_fix = self.inflation_boundary(self.obsx, self.obsy, distance=[1, 2])
        self.set_object(self.inflation_fix)
        self.inflation_1 = self.inflation_boundary(self.obsx, self.obsy, distance=[2, 3])
        self.inflation_2 = self.inflation_boundary(self.obsx, self.obsy, distance=[4, 5])

    def inflation_boundary(self, obsx, obsy, distance=[0, 1]):
        nodes = set()
        for val in zip(obsx, obsy):
            for r in range(distance[0], distance[1]):
                neighbors = get_movements_8n(x=val[0], y=val[1], r=r)
                for n in neighbors:
                    if heuristic(val, n) <= distance[1] and self.in_bounds(n) and self.is_unoccupied(n):
                        if n not in zip(obsx, obsy):
                            nodes.add(n)
        results = list(nodes)
        #results = np.array(results)
        return results

    def inflation_obstacles(self, pos, obsx, obsy, distance=[0, 1]):
        nodes = set()
        for val in zip(obsx, obsy):
            for r in range(distance[0], distance[1]):
                neighbors = get_movements_8n(x=val[0], y=val[1], r=r)
                for n in neighbors:
                    if n not in zip(obsx, obsy) and heuristic(val, n) <= distance[1]:
                        nodes.add(n)
        results = list(nodes)
        return results

    def get_map(self):
        """Return the current occupancy grid map"""
        return self.occupancy_grid_map
    
    def set_map(self, new_occgrid):
        """Set a new occupancy grid map"""
        self.occupancy_grid_map = new_occgrid
        
    def in_bounds(self, pos: (int, int)) -> bool:
        """Check if a provided pos is within the bounds of the grid map"""
        (x, y) = (round(pos[0]), round(pos[1]))
        return self.config.x_min < x < self.config.x_max and self.config.y_min < y < self.config.y_max
    
    def is_unoccupied(self, pos: (int, int)) -> bool:
        """Check if a provided pos is un_occupied"""
        (x, y) = (round(pos[0]), round(pos[1]))
        (row, col) = (int(x), int(y))
        return self.occupancy_grid_map[row, col] == UNOCCUPIED
    
    def is_uninflation(self, pos: (int, int)) -> bool:
        """Check if a provided pos is within the inflation of obstacles"""
        (x, y) = (round(pos[0]), round(pos[1]))
        return (x, y) not in self.inflation
    
    def filter(self, neighbors: List, avoid_obstacles: bool):
        """Filter neighbors"""
        if avoid_obstacles:
            return [node for node in neighbors if self.in_bounds(node) and self.is_unoccupied(node)]
        return [node for node in neighbors if self.in_bounds(node)]
    
    def get_neighbors(self, pos: (int, int), avoid_obstacles: bool = False) -> list:
        (x, y) = pos
        if self.exploration_setting == '4N':
            neighbors = get_movements_4n(x=x, y=y)
        elif self.exploration_setting == '8N':
            neighbors = get_movements_8n(x=x, y=y)
        else:
            raise ValueError ("Exploration method is not existed!")
            return None
        
        if (x + y) % 2 == 0:
            neighbors.reverse()
        
        filtered_neighbors = self.filter(neighbors=neighbors, avoid_obstacles=avoid_obstacles)
        return list(filtered_neighbors)
    
    def set_object(self, obj):
        for pos in obj:
            if self.is_unoccupied(pos):
                self.set_obstacle(pos)
        
    def remove_object(self, obj):
        for pos in obj:
            if not self.is_unoccupied(pos):
                self.remove_obstacle(pos)
        
    def set_obstacle(self, pos: (int, int)):
        """Set a provided pos being an obstacle"""
        (row, col) = (round(pos[0]), round(pos[1]))
        self.occupancy_grid_map[row, col] = OBSTACLE
    
    def remove_obstacle(self, pos: (int, int)):
        """Set a provided pos from an obstacle (occupied pos) to un_occupied pos"""
        (row, col) = (round(pos[0]), round(pos[1]))
        self.occupancy_grid_map[row, col] = UNOCCUPIED
        
    def local_observation(self, global_pos: (int, int), view_range: int = 2, detected_obs=[]) -> Dict:
        """Return a local grid map in view range of the robot: local observation"""
        (gx, gy) = global_pos[:2]
        nodes = [(x, y) for x in range(int(gx) - view_range, int(gx) + view_range +1) for y in range(int(gy) - view_range, int(gy) + view_range +1)
                 if self.in_bounds((x, y))]
        results = {node: UNOCCUPIED if self.is_unoccupied(pos=node) else OBSTACLE for node in nodes}
        if len(detected_obs) > 0:
            for node in detected_obs:
                results[tuple(node)] = OBSTACLE
        return results
    
    def local_nodes(self, global_pos: (int, int), view_range: int = 2) -> list:
        """Return a local grid map in view range of the robot: local observation"""
        (gx, gy) = global_pos
        nodes = [(x, y) for x in range(gx - view_range, gx + view_range +1) for y in range(gy - view_range, gy + view_range +1)
                 if self.in_bounds((x, y))]
        return nodes

def main():
    print("Initialization...")
    rospy.init_node("Map")
    amap = OccupancyGridMap(resolution=2)
    print(amap.occupancy_grid_map.shape)
    
    # plot
    print("Ploting...")
    plt.cla()
    plt.title("Check inflated region of the environment")
    plt.plot(amap.obsx, amap.obsy, '.k')
    plt.plot(amap.inflation[:,0], amap.inflation[:,1], '.g')
    #plt.plot(lethal_obs[:,0], lethal_obs[:,1], 'dr')
    plt.axis("equal")
    plt.grid("True")
    plt.show()
    print("Finished!")
    

if __name__ == '__main__':
    main()

