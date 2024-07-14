# tiago_gazebo
## Usage
To run the Gazebo simulation:
```bash
roslaunch tiago_gazebo tiago_gazebo.launch public_sim:=true end_effector:=pal-gripper world:=WORLD
```
where `WORLD` is one of the worlds in one of the packages `gazebo_worlds`
or `pal_gazebo_worlds`.
