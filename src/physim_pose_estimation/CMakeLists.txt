cmake_minimum_required(VERSION 2.8.3)
project(physim_pose_estimation)

set(PCL_DIR /home/chaitanya/software/pcl-pcl-1.7.2/build)
find_package (PCL REQUIRED COMPONENTS io common visualization features) 

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs 
  cv_bridge
  pcl_ros
  tf
  image_transport 
  image_geometry
  message_generation
  geometry_msgs
)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Debug)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  ObjectPose.msg
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  EstimateObjectPose.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs  # Or other packages containing msgs
  geometry_msgs
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  DEPENDS cv_bridge
  CATKIN_DEPENDS 
  image_transport
  image_geometry 
  roscpp
  pcl_ros
  tf
  message_generation
  geometry_msgs
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include)
include_directories(src/data_layer)
include_directories(src/hypothesis_generation)
include_directories(src/hypothesis_verification)
include_directories(src/hypothesis_verification/greedy_bfs)
include_directories(src/hypothesis_verification/mcts)
include_directories(src/hypothesis_verification/physics_reasoning)
include_directories(src/misc)
include_directories(src/segmentation)
include_directories(src/visualization)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${CATKIN_DEVEL_PREFIX}/include
)

## Declare a C++ executable
add_executable(physim_pose_estimation ${RIGIDBODY_EXAMPLE_OBJS}
                          src/main.cpp 
                          src/data_layer/GlobalCfg.cpp
                          src/data_layer/SceneCfg.cpp
                          src/data_layer/Objects.cpp
                          src/misc/utilities.cpp
                          src/segmentation/Segmentation.cpp
                          src/hypothesis_generation/ObjectPoseCandidateSet.cpp
                          src/hypothesis_verification/HypothesisSelection.cpp
                          src/visualization/PoseVisualization.cpp
                          )

add_executable(physim_pose_visualization
                          src/main_viz.cpp 
                          src/data_layer/GlobalCfg.cpp
                          src/data_layer/SceneCfg.cpp
                          src/data_layer/Objects.cpp
                          src/misc/utilities.cpp
                          src/segmentation/Segmentation.cpp
                          src/hypothesis_generation/ObjectPoseCandidateSet.cpp
                          src/hypothesis_verification/HypothesisSelection.cpp
                          src/visualization/PoseVisualization.cpp
                          )

add_executable(physim_pose_featureGen src/main_genModelFeatures.cpp)
                          
add_dependencies(physim_pose_estimation ${catkin_EXPORTED_TARGETS})
add_dependencies(physim_pose_visualization ${catkin_EXPORTED_TARGETS})
add_dependencies(physim_pose_featureGen ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(physim_pose_estimation
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  ${CATKIN_DEVEL_PREFIX}/lib/libsuper4pcs.so
)

target_link_libraries(physim_pose_visualization
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  ${CATKIN_DEVEL_PREFIX}/lib/libsuper4pcs.so
)

target_link_libraries(physim_pose_featureGen
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  ${CATKIN_DEVEL_PREFIX}/lib/libsuper4pcs.so
)