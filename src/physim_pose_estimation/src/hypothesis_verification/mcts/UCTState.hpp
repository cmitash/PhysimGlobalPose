#ifndef UCT_STATE
#define UCT_STATE

#include <APCObjects.hpp>
#include <PhySim.hpp>

namespace uct_state{
	
	class UCTState{
		public:
			UCTState(unsigned int numObjects, int numChildNodes, UCTState* parent);
			~UCTState();
			void copyParent(UCTState*);
			void updateNewObject(apc_objects::APCObjects*, std::pair <Eigen::Isometry3d, float>, int maxDepth);
			void render(Eigen::Matrix4f, std::string);
			void updateStateId(int num);
			void computeCost(cv::Mat obsImg);
			void performTrICP(std::string scenePath, float trimPercentage);
			void correctPhysics(physim::PhySim*, Eigen::Matrix4f, std::string);
			UCTState* getBestChild(std::string scenePath);
			bool isFullyExpanded();
			void updateChildHval(std::vector< std::pair <Eigen::Isometry3d, float> > childStates);

			std::string stateId;
			unsigned int numObjects;
			int numChildren;

			std::vector<std::pair<apc_objects::APCObjects*, Eigen::Isometry3d> > objects;
			UCTState* parentState;
			std::vector<UCTState*> children;
			std::vector<int> isExpanded;
			std::vector<float> hval;

			cv::Mat renderedImg;
			int numExpansions;
			unsigned int renderScore;
			float qval;
	};
}// namespace

#endif