#ifndef SEARCH
#define SEARCH

#include <State.hpp>
#include <Scene.hpp>

namespace search{
	
	class Search{
		public:
			Search(scene::Scene *currScene);
			void dfsSearch();
			void expandNode();

			state::State *rootState;
			std::vector<apc_objects::APCObjects*> objOrder;
			
	};
}// namespace

#endif