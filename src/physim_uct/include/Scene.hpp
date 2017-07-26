#ifndef SCENE
#define SCENE

#include <common_io.h>
#include <APCObjects.hpp>

namespace scene{

	class Scene{
		public:
			Scene(std::string scenePath);
			void performRCNNDetection();

			int numObjects;
			std::string scenePath;
			cv::Mat color_image, depth_image;
			std::vector<apc_objects::APCObjects*> sceneObjs;
			Eigen::Matrix4f camPose;
			Eigen::Matrix3f camIntrinsic;
	};
}//namespace
#endif