## PhysimGlobalPose
This repository implements a search-based technique for 6D pose estimation of objects in clutter as described in our paper.
#### A Self-supervised Learning System for Object Detection using Physics Simulation and Multi-view Pose Estimation ([pdf](https://arxiv.org/abs/1710.08577))([website](http://paul.rutgers.edu/~cm1074/research/icra18/MCTS.html))
By Chaitanya Mitash, Kostas Bekris, Abdeslam Boularias (Rutgers University).
In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), Brisbane, Australia, 2018.

### Setup
1. Clone the repository.
2. Download object models: [Models](https://drive.google.com/drive/folders/1VvIpDOrYlZJ-opGyhXf9o4xBg39lSL3o?usp=sharing)
3. Copy models to ```$PHYSIM_GLOBAL_POSE/src/physim_pose_estimation/```
4. Download trained fcn weights: [FCN Model](https://drive.google.com/drive/folders/1U-JI5SZhA1qwk-gfc2t9dxV56X6eTwWx?usp=sharing)
5. Copy weights to ```$PHYSIM_GLOBAL_POSE/src/3rdparty/fcn_segmentation_package```
6. Download and extract [Bullet](https://github.com/bulletphysics/bullet3/releases/tag/2.86.1)

### Demo
```
export BULLET_PHYSICS_PATH=/path/to/bullet/bullet3-2.86.1/
export PHYSIM_GLOBAL_POSE=/path/to/repo/PhysimGlobalPose
source $PHYSIM_GLOBAL_POSE/devel/setup.sh

cd $PHYSIM_GLOBAL_POSE/src
catkin_init_workspace
cd $PHYSIM_GLOBAL_POSE
catkin_make
rosrun physim_pose_estimation physim_pose_estimation
run $PHYSIM_GLOBAL_POSE/src/3rdparty/fcn_segmentation_package/predict
rosservice call /pose_estimation "APC" "$PHYSIM_GLOBAL_POSE/test-scene/" "FCNThreshold" "PCS" "LCP"
```

### Output
1. Estimated 6D pose of all objects in the scene.

### System Requirements
1. Ubuntu 14.04/16.04
2. Cuda 8.0, CudNN 5.0

### Citing
To cite the work:

```
@inproceedings{mitash2017improving,
  Author = {Mitash, Chaitanya and Boularias, Abdeslam and Bekris, Kostas E},
  Booktitle = {{IEEE} International Conference on Robotics and Automation (ICRA)},
  Title = {Improving 6D Pose Estimation of Objects in Clutter via Physics-aware Monte Carlo Tree Search},
  Year = {2018}}
```