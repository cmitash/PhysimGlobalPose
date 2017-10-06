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
			void performLCCPSegmentation();
			void getOrder();
			void readHypothesis();

			void kernelKMeans(cv::Mat &rotPoints, cv::Mat &rotScores, Eigen::Vector3f symInfo, cv::Mat &rotCenters,
				cv::Mat &rotCenterScores);

			void performKernelKMeansRotation(std::vector<cv::Mat> &transClusters, std::vector<cv::Mat> &transScores, 
				cv::Mat& transCenters, apc_objects::APCObjects* obj, cv::Mat &clusterReps, cv::Mat &allClusterScores);

			void performKMeansTranslation(cv::Mat points, cv::Mat scores, std::vector<cv::Mat> &transClusters, 
				std::vector<cv::Mat> &scoreTrans, cv::Mat& transCenters, apc_objects::APCObjects* obj);

			void performKMeans(cv::Mat points, cv::Mat &clusterIndices, cv::Mat &clusterCenters, 
				apc_objects::APCObjects* obj, std::vector< std::pair <Eigen::Isometry3d, float> >& subsetPose);

			void getHypothesis(apc_objects::APCObjects* obj, std::pair <Eigen::Isometry3d, float> &bestLCPPose, 
				std::vector< std::pair <Eigen::Isometry3d, float> > &allSuperPCSposes);

			void descretizeHypothesisSet(apc_objects::APCObjects* obj, float bestscore, std::map<std::string, float> &poseMap, 
				std::vector< std::pair <Eigen::Isometry3d, float> > &allSuperPCSposes);

			void computeHypothesisSet();

			void clusterHypothesisSet(apc_objects::APCObjects* obj, std::map<std::string, float> &poseMap,
				std::vector< std::pair <Eigen::Isometry3d, float> > &clusteredPoses);

			
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