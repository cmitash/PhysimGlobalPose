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
					std::string scenePath, Eigen::Matrix4f camPose, cv::Mat depthImage, std::vector<float> cutOffScore);
			void performSearch();
			uct_state::UCTState* expand(uct_state::UCTState *currState);
			uct_state::UCTState * treePolicy(uct_state::UCTState *currState);
			float defaultPolicy(uct_state::UCTState *selState);
			void backupReward(uct_state::UCTState *selState, float reward);

			uct_state::UCTState *rootState;
			
			std::vector<apc_objects::APCObjects*> objOrder;
			std::vector< std::vector< std::pair <Eigen::Isometry3d, float> > > unconditionedHypothesis;
			std::string scenePath;
			Eigen::Matrix4f camPose;
			cv::Mat depthImage;
			std::vector<float> cutOffScore;

			uct_state::UCTState *bestState;
			unsigned int bestScore;

			physim::PhySim *pSim;
			std::priority_queue<uct_state::UCTState*> pq;

	};
}// namespace

#endif