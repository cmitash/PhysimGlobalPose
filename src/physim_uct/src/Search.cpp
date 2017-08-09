#include <Search.hpp>

bool operator<(const state::State& lhs, const state::State& rhs) {
	return lhs.hval < rhs.hval;
}

namespace search{
	
	/********************************* function: constructor ***********************************************
	*******************************************************************************************************/

	Search::Search(scene::Scene *currScene){
		rootState = new state::State(0);
		rootState->updateStateId(0);

		this->currScene = currScene;
		this->objOrder = currScene->objOrder;
		this->unconditionedHypothesis = currScene->unconditionedHypothesis;

		// initialize physics engine
		pSim = new physim::PhySim();
		pSim->addTable(0.53);
		for(int ii=0; ii<objOrder.size(); ii++)
			pSim->initRigidBody(objOrder[ii]->objName);
	}

	/********************************* function: expandNode ***********************************************
	*******************************************************************************************************/

	void Search::expandNode(state::State* expState){
		expState->expand();
		expState->performTrICP(currScene->scenePath);
		expState->correctPhysics(pSim, currScene->camPose, currScene->scenePath);

		unsigned int maxDepth = objOrder.size();
		if(expState->numObjects == maxDepth){
			cv::Mat depth_image;
			expState->render(currScene->camPose, currScene->scenePath, depth_image);
			expState->computeCost(depth_image, currScene->depthImage);
		}
		else{
			unsigned int nextDepthLevel = expState->numObjects + 1;
			for(int ii = 0; ii < unconditionedHypothesis[nextDepthLevel - 1].size(); ii++){
				state::State* childState = new state::State(nextDepthLevel);
				childState->copyParent(expState);
				childState->updateStateId(ii);
				childState->updateNewObject(objOrder[nextDepthLevel - 1], unconditionedHypothesis[nextDepthLevel - 1][ii], maxDepth);
				pq.push(childState);
			}
		}
	}

	/********************************* function: heuristicSearch *******************************************
	*******************************************************************************************************/

	void Search::heuristicSearch(){
		const clock_t begin_time = clock();
		
		pq.push(rootState);
		while(!pq.empty()){
			
			if((float( clock () - begin_time ) /  CLOCKS_PER_SEC) > 1)
				break;
			
			state::State *expState = pq.top();
			pq.pop();
			expandNode(expState);
		}
	}

	/********************************* end of functions ****************************************************
	*******************************************************************************************************/
}