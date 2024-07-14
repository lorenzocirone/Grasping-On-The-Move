# AMR23-FP1-UkfSlipping


## Installation

Make sure you have followed the installation instructions in [http://wiki.ros.org/Robots/TIAGo/Tutorials](http://wiki.ros.org/Robots/TIAGo/Tutorials), either installing directly on Ubuntu 20.04 or through Docker. Install catkin_tools, create a catkin workspace and clone this repository in the `src` folder. 

Before compiling you have to source the tiago workspace whenever is located
```
source <PATH_2_TIAGO_WS>/tiago_public_ws/devel/setup.bash
```

Make sure you are compiling in *Release* mode by properly setting your catkin workspace:
```bash
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release
```
Build your code by running the following command:
```bash
catkin build
```
note that if you are using python it is sufficient to build the package only once

After the build is completed for the first time, you have to *source the current setup.bash* that have been created into the `devel` folder
```
source ./devel/setup.bash
```

## Usage

### On the move simulations

First run the Gazebo world:
```bash
roslaunch tiago_gazebo tiago_gazebo.launch public_sim:=true end_effector:=pal-gripper world:=post_ball_polygonbox
```

Then run the node to execute simoultaneously the base and the arm controller
```bash
roslaunch tiago_grasping_on_the_move tiago_grasping_on_the_move.launch
```

### Starting the simulation and loading the world

To run the Gazebo simulation:
```bash
roslaunch tiago_gazebo tiago_gazebo.launch public_sim:=true end_effector:=pal-gripper world:=<WORLD>     
```
Where `<WORLD>` is one of the worlds in `gazebo_worlds/worlds`.

For example, run 
```bash
roslaunch tiago_gazebo tiago_gazebo.launch public_sim:=true end_effector:=pal-gripper world:=empty     
```

### Run the simulation

To run the written module:
```bash
roslaunch tiago_grasping_on_the_move tiago_grasping_on_the_move.launch
```