#ifndef COMMON_IO
#define COMMON_IO

#include <iostream>
#include <string>
#include <map>
#include <vector>

#include <ros/ros.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// definitions
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudRGB;

// Global variables
extern std::string env_p;

#endif