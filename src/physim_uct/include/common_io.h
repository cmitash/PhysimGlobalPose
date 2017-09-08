#ifndef COMMON_IO
#define COMMON_IO

// STL
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <cmath>
#include <time.h>
#include <queue>
#include <climits>
#include <boost/assign.hpp>

// Basic ROS
#include <ros/ros.h>

// Basic PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>

// For IO
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>

// For OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// For Visualization
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

// For RANSAC
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/sac_model_plane.h>

// For ICP
#include <pcl/registration/icp.h>
#include <pcl/recognition/ransac_based/auxiliary.h>
#include <pcl/recognition/ransac_based/trimmed_icp.h>

#include <geometry_msgs/Pose.h>

// definitions
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudRGB;

// Global variables
extern std::string env_p;
extern std::map<std::string, Eigen::Vector3f> symMap; // FIX ME

#define DBG_SUPER4PCS
// #define DBG_ICP
// #define DBG_PHYSICS
#define DGB_RESULT

// Declaration for common utility functions
namespace utilities{
	std::string type2str(int type);
	void convert3dOrganized(cv::Mat &objDepth, Eigen::Matrix3f &camIntrinsic, PointCloud::Ptr objCloud);
	void convert3dUnOrganized(cv::Mat &objDepth, Eigen::Matrix3f &camIntrinsic, PointCloud::Ptr objCloud);
	boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud);
	void readDepthImage(cv::Mat &depthImg, std::string path);
	void writeDepthImage(cv::Mat &depthImg, std::string path);
	void convert2d(cv::Mat &objDepth, Eigen::Matrix3f &camIntrinsic, PointCloud::Ptr objCloud);
	void TransformPolyMesh(const pcl::PolygonMesh::Ptr &mesh_in, pcl::PolygonMesh::Ptr &mesh_out, Eigen::Matrix4f transform);
	void convertToMatrix(Eigen::Isometry3d &from, Eigen::Matrix4f &to);
	void convertToIsometry3d(Eigen::Matrix4f &from, Eigen::Isometry3d &to);
	void convertToWorld(Eigen::Matrix4f &transform, Eigen::Matrix4f &cam_pose);
	void convertToCamera(Eigen::Matrix4f &tform, Eigen::Matrix4f &cam_pose);
	float getRotDistance(Eigen::Matrix3f rotMat1, Eigen::Matrix3f rotMat2, Eigen::Vector3f symInfo);
	void getPoseError(Eigen::Matrix4f testPose, Eigen::Matrix4f gtPose, Eigen::Vector3f symInfo, 
		float &meanrotErr, float &transErr);
	void getEMDError(Eigen::Matrix4f testPose, Eigen::Matrix4f gtPose, PointCloud::Ptr objModel, float &error,
		std::pair<float, float> &xrange, std::pair<float, float> &yrange, std::pair<float, float> &zrange);
	void convertToCVMat(Eigen::Matrix4f &pose, cv::Mat &cvPose);
	void convert6DToMatrix(Eigen::Matrix4f &pose, cv::Mat &points, int index);
	void toQuaternion(Eigen::Vector3f& eulAngles, Eigen::Quaternionf& q);
	Eigen::Vector3f rotationMatrixToEulerAngles(Eigen::Matrix3f R);
	void writePoseToFile(Eigen::Matrix4f pose, std::string objName, std::string scenePath, std::string filename);
	void writeScoreToFile(float score, std::string objName, std::string scenePath, std::string filename);
}

#endif