#ifndef STATE
#define STATE

#include <APCObjects.hpp>

namespace state{
	
	class State{
		public:
			State(unsigned int numObjects);
			void expand();
			void copyParent(State*);
			void updateNewObject(apc_objects::APCObjects*, std::pair <Eigen::Isometry3d, float>, int maxDepth);
			void render(Eigen::Matrix4f, std::string, cv::Mat &depth_image);
			void updateStateId(int num);
			void computeCost(cv::Mat renderedImg, cv::Mat obsImg);

			std::string stateId;
			unsigned int numObjects;
			std::vector<std::pair<apc_objects::APCObjects*, Eigen::Isometry3d> > objects;
			float hval;
			unsigned int score;
	};
}// namespace

#endif