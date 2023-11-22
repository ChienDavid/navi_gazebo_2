#############################
### Path messages
#############################
### estimated pose of robot
float32 robot_x
float32 robot_y
float32 robot_theta

### global path
float32[] globalCoordXs
float32[] globalCoordYs

### local path
float32[] localCoordXs
float32[] localCoordYs

### subtarget
int32 subTarget_x_sim
int32 subTarget_y_sim
float32 subTarget_x_global
float32 subTarget_y_global
float32 subTarget_x_local
float32 subTarget_y_local
float32 subTarget_theta_local

