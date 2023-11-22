Indoor navigation of multi robots in Gazebo simulation



Commands: run each command on different terminals
- roscore
- roslaunch turtlebot3_gazebo multi_turtlebot3_world.launch
- rosrun navi_gazebo robot_obs.py
- roslaunch navi_gazebo robot0.launch
- roslaunch navi_gazebo robot1.launch



Given: a repository "multi robot gazebo navistack"
- Indoor map of the environment
- Two mobile robot (Turtlebot 3)
- LiDAR sensor
- Localization



Task: two robots autonomously navigate to different target locations without collision even if it is commanded to cross.
- A target location is randomly selected.
- The robot has to avoid fixed obstacles (walls, cabinets, tables).
- Robots have to avoid each other.



Solution:
- Global path planner: A* algorithm
- Path tracking controller: Pure pursuit
- Motion prediction



Experimental environment:
- OS (in Virtualbox): Ubuntu 20.04
- Middleware and software: ROS Noetic
- Programming language: Python (version 3x)



Video:
https://youtu.be/VZaxRao7CzQ



References:
- A* algorithm: P.E. Hart, N.J. Nilsson and B. Raphael, “A formal basis for the heuristic determination of minimum cost paths,” IEEE Transactions on Systems Science and Cybernetics, vol. 4, no. 2, pp. 100–107, 1968.
- Pure pursuit controller: Jarrod M. Snider, "Automatic Steering Methods for Autonomous Automobile Path Tracking", 2009
- PythonRobotics: https://github.com/AtsushiSakai/PythonRobotics/tree/master

