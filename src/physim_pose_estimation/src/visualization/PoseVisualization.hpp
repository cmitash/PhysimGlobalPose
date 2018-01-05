#ifndef POSE_VISUALIZATION
#define POSE_VISUALIZATION

#include <common_io.h>

namespace pose_visualization{
	
	class PoseVisualization{
	public:
		PoseVisualization();
		~PoseVisualization();

		void loadSceneCloud(cv::Mat depthImage, cv::Mat colorImage, Eigen::Matrix3f camIntrinsic, Eigen::Matrix4f camPose);
		void startViz();
		void loadObjectModels(pcl::PolygonMesh meshObj, std::vector<double> poseObj, std::string meshName);

		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	};
}

#endif