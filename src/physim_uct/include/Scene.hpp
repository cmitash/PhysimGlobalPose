#ifndef SCENE
#define SCENE

#include <common_io.h>
#include <APCObjects.hpp>

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
			#endif

			void getUnconditionedHypothesis();
			
			int numObjects;
			std::string scenePath;
			cv::Mat colorImage, depthImage;
			PointCloud::Ptr sceneCloud;
			std::vector<apc_objects::APCObjects*> sceneObjs;
			std::vector<apc_objects::APCObjects*> objOrder;
			std::vector< std::vector< std::pair <Eigen::Isometry3d, float> > > unconditionedHypothesis;
			std::vector< std::pair<Eigen::Isometry3d, float> > max4PCSPose;
			Eigen::Matrix4f camPose;
			Eigen::Matrix3f camIntrinsic;

			#ifdef DBG_SUPER4PCS
			std::vector< std::pair <apc_objects::APCObjects*, Eigen::Matrix4f> > groundTruth;
			void readGroundTruth();
			void getHypothesis(apc_objects::APCObjects* obj, PointCloud::Ptr pclSegment, PointCloud::Ptr pclModel, 
				std::vector< std::pair <Eigen::Isometry3d, float> > &allPose, Eigen::Matrix4f gtPose);
			#endif
	};
}//namespace
#endif