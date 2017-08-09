#ifndef SEARCH
#define SEARCH

#include <State.hpp>
#include <Scene.hpp>
#include <PhySim.hpp>

namespace search{
	
	class Search{
		public:
			Search(scene::Scene *currScene);
			void heuristicSearch();
			void expandNode(state::State*);

			state::State *rootState;
			scene::Scene *currScene;
			std::vector<apc_objects::APCObjects*> objOrder;
			physim::PhySim *pSim;
			state::State *bestState;
			unsigned int bestScore;
			
			std::priority_queue<state::State*> pq;
			std::vector< std::vector< std::pair <Eigen::Isometry3d, float> > > unconditionedHypothesis;
	};
}// namespace

#endif