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
#include <thread>

// Basic ROS
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>

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
#include <visualization_msgs/Marker.h>

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

// For normal estimation
#include <pcl/features/normal_3d.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/surface/mls.h>

#include <geometry_msgs/Pose.h>

#include <pointmatcher/PointMatcher.h>

// definitions
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudRGB;
typedef pcl::PointCloud<pcl::PointXYZRGBNormal> PointCloudRGBNormal;

typedef PointMatcher<float> PM;
typedef PM::DataPoints DP;
typedef PM::Parameters Parameters;

// #define DBG_ICP
// #define DBG_PHYSICS

// Declaration for common utility functions
namespace utilities{
	// global variable
	extern std::map<std::string, geometry_msgs::Pose> anyTimePoseArray;
	extern PointCloudRGB::Ptr pc_viz;

	std::string type2str(int type);
	void convert3dOrganized(cv::Mat &objDepth, Eigen::Matrix3f &camIntrinsic, PointCloud::Ptr objCloud);
	void convert3dOrganizedRGB(cv::Mat &objDepth, cv::Mat &colImage, Eigen::Matrix3f &camIntrinsic, PointCloudRGB::Ptr objCloud);
	void convert3dUnOrganized(cv::Mat &objDepth, Eigen::Matrix3f &camIntrinsic, PointCloud::Ptr objCloud);
	void convert3dUnOrganizedRGB(cv::Mat &objDepth, cv::Mat &colorImage, Eigen::Matrix3f &camIntrinsic, PointCloudRGB::Ptr objCloud);
	boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud);
	void readDepthImage(cv::Mat &depthImg, std::string path);
	void readProbImage(cv::Mat &probImg, std::string path);
	void writeDepthImage(cv::Mat &depthImg, std::string path);
	void writeClassImage(cv::Mat &classImg, cv::Mat colorImage, std::string path);
	void convert2d(cv::Mat &objDepth, Eigen::Matrix3f &camIntrinsic, PointCloud::Ptr objCloud);
	void TransformPolyMesh(const pcl::PolygonMesh::Ptr &mesh_in, pcl::PolygonMesh::Ptr &mesh_out, Eigen::Matrix4f transform);
	void convertToMatrix(Eigen::Isometry3d &from, Eigen::Matrix4f &to);
	void convertToIsometry3d(Eigen::Matrix4f &from, Eigen::Isometry3d &to);
	void invertTransformationMatrix(Eigen::Matrix4f &tform);
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
	void toTransformationMatrix(Eigen::Matrix4f& camPose, std::vector<double> camPose7D);
	void performTrICP(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclSegment, 
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclModel, 
		Eigen::Isometry3d &currTransform,
		Eigen::Isometry3d &finalTransform,
		float trimPercentage);
	void pointToPlaneICP(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclSegment, 
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclModel, 
		Eigen::Matrix4f &offsetTransform);
	void pointMatcherICP(std::string refPath, 
		std::string dataPath, 
		Eigen::Matrix4f &offsetTransform);
}

#endif