#ifndef SCENE
#define SCENE

#include <common_io.h>
#include <APCObjects.hpp>
#include <State.hpp>

namespace scene{
	
	class Scene{
		public:
			Scene(std::string scenePath);
			void performRCNNDetection();
			void get3DSegments();
			void removeTable();
			void getOrder();

			void getHypothesis(apc_objects::APCObjects* obj, PointCloud::Ptr pclSegment, PointCloud::Ptr pclModel);

			void clusterPoseSet(cv::Mat points, cv::Mat &clusterIndices, cv::Mat &clusterCenters, apc_objects::APCObjects* obj,
									 std::vector< std::pair <Eigen::Isometry3d, float> >& subsetPose);

			void clusterTransPoseSet(cv::Mat points, cv::Mat scores, std::vector<cv::Mat> &transClusters, std::vector<cv::Mat> &,
										 cv::Mat& transCenters, apc_objects::APCObjects* obj);

			void clusterRotWithinTrans(std::vector<cv::Mat> &transClusters, std::vector<cv::Mat> &, cv::Mat& transCenters,
											apc_objects::APCObjects* obj, std::vector< std::pair <Eigen::Isometry3d, float> >& subsetPose);

			void kernelKMeans(cv::Mat &rotPts,cv::Mat &rotCenters, Eigen::Vector3f symInfo);

			void getUnconditionedHypothesis();
			
			int numObjects;
			std::string scenePath;
			cv::Mat colorImage, depthImage;
			PointCloud::Ptr sceneCloud;
			std::vector<apc_objects::APCObjects*> sceneObjs;
			std::vector<apc_objects::APCObjects*> objOrder;
			std::vector<std::vector<apc_objects::APCObjects*> > independentTrees;
			std::vector< std::vector< std::pair <Eigen::Isometry3d, float> > > unconditionedHypothesis;
			
			Eigen::Matrix4f camPose;
			Eigen::Matrix3f camIntrinsic;
			
			std::vector<float> cutOffScore;
			std::vector< std::pair<Eigen::Isometry3d, float> > max4PCSPose;
			
			state::State* finalState;
	};
}//namespace
#endif