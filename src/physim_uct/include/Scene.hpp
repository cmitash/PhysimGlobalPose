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

			#ifndef DBG_SUPER4PCS
			void getHypothesis(apc_objects::APCObjects* obj, PointCloud::Ptr pclSegment, PointCloud::Ptr pclModel, 
				std::vector< std::pair <Eigen::Isometry3d, float> > &allPose);
			#else
			void getHypothesis(apc_objects::APCObjects* obj, PointCloud::Ptr pclSegment, PointCloud::Ptr pclModel, 
				std::vector< std::pair <Eigen::Isometry3d, float> > &allPose, Eigen::Matrix4f gtPose);
			#endif
			void clusterPoseSet(cv::Mat points, cv::Mat &clusterIndices, cv::Mat &clusterCenters,
								int &bestClusterIdx, apc_objects::APCObjects* obj, Eigen::Matrix4f gtPose,
			 					int k, std::vector< std::pair <Eigen::Isometry3d, float> >& subsetPose);
			void customClustering(cv::Mat points, cv::Mat &clusterIndices, cv::Mat &clusterCenters,
								int &bestClusterIdx, apc_objects::APCObjects* obj, Eigen::Matrix4f gtPose,
			 					int k, std::vector< std::pair <Eigen::Isometry3d, float> >& subsetPose);
			void withinCLusterLookup(cv::Mat points, apc_objects::APCObjects* obj, Eigen::Matrix4f gtPose,
			 						cv::Mat clusterIndices, int bestClusterIdx);
			void clusterTransPoseSet(cv::Mat points, cv::Mat scores, std::vector<cv::Mat> &transClusters, std::vector<cv::Mat> &,
										 cv::Mat& transCenters, apc_objects::APCObjects* obj, int k);
			void clusterRotWithinTrans(std::vector<cv::Mat> &transClusters, std::vector<cv::Mat> &, cv::Mat& transCenters, int k,
									  apc_objects::APCObjects* obj, std::vector< std::pair <Eigen::Isometry3d, float> >& subsetPose, Eigen::Matrix4f gtPose);
			void getUnconditionedHypothesis();
			
			int numObjects;
			std::string scenePath;
			cv::Mat colorImage, depthImage;
			PointCloud::Ptr sceneCloud;
			std::vector<apc_objects::APCObjects*> sceneObjs;
			std::vector<apc_objects::APCObjects*> objOrder;
			std::vector<std::vector<apc_objects::APCObjects*> > independentTrees;
			std::vector< std::vector< std::pair <Eigen::Isometry3d, float> > > unconditionedHypothesis;
			std::vector< std::pair<Eigen::Isometry3d, float> > max4PCSPose;
			std::vector<float> cutOffScore;
			Eigen::Matrix4f camPose;
			Eigen::Matrix3f camIntrinsic;
			float lcpThreshold;
			state::State* finalState;
			std::vector<std::vector<cv::Mat> > clusters;
			std::vector<std::vector<cv::Mat> > clusterScores;

			std::vector< std::pair <apc_objects::APCObjects*, Eigen::Matrix4f> > groundTruth;
			void readGroundTruth();
	};
}//namespace
#endif