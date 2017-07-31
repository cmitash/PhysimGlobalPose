#include <Search.hpp>

namespace search{
	
	Search::Search(scene::Scene *currScene){
		rootState = new state::State(0);
		objOrder = currScene->getOrder();
	}

	void Search::dfsSearch(){

	}
}