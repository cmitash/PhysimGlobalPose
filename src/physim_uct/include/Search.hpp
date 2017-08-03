#ifndef SEARCH
#define SEARCH

#include <State.hpp>
#include <Scene.hpp>

namespace search{
	
	class Search{
		public:
			Search(scene::Scene *currScene);
			void heuristicSearch();
			void expandNode(state::State*);

			state::State *rootState;
			scene::Scene *currScene;
			std::vector<apc_objects::APCObjects*> objOrder;
			
			std::priority_queue<state::State*> pq;
			std::vector< std::vector< std::pair <Eigen::Isometry3d, float> > > unconditionedHypothesis;
	};
}// namespace

#endif