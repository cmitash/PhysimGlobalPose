#include <Search.hpp>

bool operator<(const state::State& lhs, const state::State& rhs) {
	return lhs.hval < rhs.hval;
}

int numExpansions;
int numRenders;
float expansionTime;
float renderTime;

namespace search{
	
	/********************************* function: constructor ***********************************************
	*******************************************************************************************************/

	Search::Search(scene::Scene *currScene){
		rootState = new state::State(0);
		rootState->updateStateId(0);

		this->currScene = currScene;
		this->objOrder = currScene->objOrder;
		this->unconditionedHypothesis = currScene->unconditionedHypothesis;

		// initialize best state
		bestState = rootState;
		bestScore = INT_MAX;

		// initialize physics engine
		pSim = new physim::PhySim();
		pSim->addTable(0.55);
		for(int ii=0; ii<objOrder.size(); ii++)
			pSim->initRigidBody(objOrder[ii]->objName);
	}

	/********************************* function: expandNode ***********************************************
	*******************************************************************************************************/

	void Search::expandNode(state::State* expState){
		expState->expand();

		const clock_t exp_begin_time = clock();
		expState->performTrICP(currScene->scenePath, 0.9);
		expState->correctPhysics(pSim, currScene->camPose, currScene->scenePath);
		expansionTime += (float( clock () - exp_begin_time ) /  CLOCKS_PER_SEC);

		unsigned int maxDepth = objOrder.size();
		if(expState->numObjects == maxDepth){
			cv::Mat depth_image;

			const clock_t render_begin_time = clock();
			expState->render(currScene->camPose, currScene->scenePath, depth_image);
			numRenders++;
			expState->computeCost(depth_image, currScene->depthImage);
			renderTime += (float( clock () - render_begin_time ) /  CLOCKS_PER_SEC);

			if(expState->score < bestScore){
				bestState = expState;
				bestScore = expState->score;
			}
		}
		else{
			unsigned int nextDepthLevel = expState->numObjects + 1;
			for(int ii = 0; ii < unconditionedHypothesis[nextDepthLevel - 1].size(); ii++){
				if(unconditionedHypothesis[nextDepthLevel - 1][ii].second > 0.8*currScene->max4PCSPose[nextDepthLevel - 1].second){
					state::State* childState = new state::State(nextDepthLevel);
					childState->copyParent(expState);
					childState->updateStateId(ii);
					childState->updateNewObject(objOrder[nextDepthLevel - 1], unconditionedHypothesis[nextDepthLevel - 1][ii], maxDepth);
					pq.push(childState);
				}
			}
		}
	}

	/********************************* function: heuristicSearch *******************************************
	*******************************************************************************************************/

	void Search::heuristicSearch(){
		const clock_t begin_time = clock();
		
		numExpansions = 0;
		numRenders = 0;
		expansionTime = 0;
		renderTime = 0;

		pq.push(rootState);
		while(!pq.empty()){
			
			if((float( clock () - begin_time ) /  CLOCKS_PER_SEC) > 30)
				break;
			
			state::State *expState = pq.top();
			pq.pop();
			expandNode(expState);
			numExpansions++;
		}
		std::cout<<"total number of state expansions: " << numExpansions <<std::endl;
		std::cout<<"mean expansion time: " << float(expansionTime/numExpansions) <<std::endl;
		std::cout<<"mean render time: " << float(renderTime/numRenders) <<std::endl;
	}

	/********************************* end of functions ****************************************************
	*******************************************************************************************************/
}