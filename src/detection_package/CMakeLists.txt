cmake_minimum_required(VERSION 2.8.3)
project(detection_package)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  message_generation
  rospy
  sensor_msgs
  cv_bridge
  std_msgs
)

catkin_python_setup()

## Generate services in the 'srv' folder
add_service_files(
  FILES
  UpdateBbox.srv
  UpdateActiveListFrame.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs  # Or other packages containing msgs
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(DIRECTORY include/detection_package/
 DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
 FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

catkin_install_python(PROGRAMS bin/detect_bbox
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})



