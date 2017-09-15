#ifndef UCT_SEARCH
#define UCT_SEARCH

#include <UCTState.hpp>
#include <Scene.hpp>
#include <PhySim.hpp>

namespace uct_search{
	
	class UCTSearch{
		public:
			UCTSearch(std::vector<apc_objects::APCObjects*> objOrder, 
					std::vector< std::vector< std::pair <Eigen::Isometry3d, float> > > unconditionedHypothesis,
					std::string scenePath, Eigen::Matrix4f camPose, cv::Mat depthImage, std::vector<float> cutOffScore, int rootId);
			~UCTSearch();
			void performSearch();
			uct_state::UCTState* expand(uct_state::UCTState *currState);
			uct_state::UCTState * treePolicy(uct_state::UCTState *currState);
			float defaultPolicy(uct_state::UCTState *selState);
			void backupReward(uct_state::UCTState *selState, float reward);
			float LCPPolicy(uct_state::UCTState *selState);

			uct_state::UCTState *rootState;

			std::vector<apc_objects::APCObjects*> objOrder;
			std::vector< std::vector< std::pair <Eigen::Isometry3d, float> > > unconditionedHypothesis;
			std::string scenePath;
			Eigen::Matrix4f camPose;
			cv::Mat depthImage;
			std::vector<float> cutOffScore;

			uct_state::UCTState *bestState;
			unsigned int bestRenderScore;

			physim::PhySim *pSim;
			std::vector<uct_state::UCTState* > allStatePtrs;
	};
}// namespace

#endif