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
			std::vector<apc_objects::APCObjects*> getOrder();
			void getHypothesis(PointCloud::Ptr pclSegment, PointCloud::Ptr pclModel, 
				std::vector< std::pair <Eigen::Isometry3d, float> > &allPose);
			void getUnconditionedHypothesis(std::vector< std::vector< std::pair <Eigen::Isometry3d, float> > > &unconditionedHypothesis,
											std::vector<apc_objects::APCObjects*> objOrder);
			
			int numObjects;
			std::string scenePath;
			cv::Mat colorImage, depthImage;
			PointCloud::Ptr sceneCloud;
			std::vector<apc_objects::APCObjects*> sceneObjs;
			Eigen::Matrix4f camPose;
			Eigen::Matrix3f camIntrinsic;
	};
}//namespace
#endif