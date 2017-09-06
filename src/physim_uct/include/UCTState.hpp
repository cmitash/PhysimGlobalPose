#ifndef UCT_STATE
#define UCT_STATE

#include <APCObjects.hpp>
#include <PhySim.hpp>

namespace uct_state{
	
	class UCTState{
		public:
			UCTState(unsigned int numObjects, std::vector< std::pair <Eigen::Isometry3d, float> > hypSet, UCTState* parent);
			void copyParent(UCTState*);
			void updateNewObject(apc_objects::APCObjects*, std::pair <Eigen::Isometry3d, float>, int maxDepth);
			void render(Eigen::Matrix4f, std::string, cv::Mat &depth_image);
			void updateStateId(int num);
			void computeCost(cv::Mat renderedImg, cv::Mat obsImg);
			void performICP(std::string scenePath, float max_corr);
			void performTrICP(std::string scenePath, float trimPercentage);
			void correctPhysics(physim::PhySim*, Eigen::Matrix4f, std::string);
			UCTState* getBestChild();
			bool isFullyExpanded();

			std::string stateId;
			unsigned int numObjects;
			std::vector<std::pair<apc_objects::APCObjects*, Eigen::Isometry3d> > objects;
			float hval;
			unsigned int score;
			int numExpansions;
			int numChildren;
			UCTState* parentState;
			std::vector<UCTState*> children;
			std::vector<int> isExpanded;
	};
}// namespace

#endif