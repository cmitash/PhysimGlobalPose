cmake_minimum_required(VERSION 2.8.3)
project(physim_uct)

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
)
find_package(OpenCV REQUIRED)


#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Debug)

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
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${CATKIN_DEVEL_PREFIX}/include
)

MESSAGE("yo" ${CATKIN_DEVEL_PREFIX}/include)

## Declare a C++ executable
add_executable(physim_uct src/physim_uct.cpp)
add_dependencies(physim_uct ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(physim_uct
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
)