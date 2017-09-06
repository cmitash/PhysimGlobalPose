#include <UCTSearch.hpp>

bool operator<(const uct_state::UCTState& lhs, const uct_state::UCTState& rhs) {
	return lhs.hval < rhs.hval;
}

namespace uct_search{
	int numExpansions;
	int numRenders;
	float expansionTime;
	float renderTime;
	clock_t search_begin_time;
	float tableHeight = 0.545;
	float trimICPthreshold = 0.9;
	
	/********************************* function: constructor ***********************************************
	*******************************************************************************************************/

	UCTSearch::UCTSearch(std::vector<apc_objects::APCObjects*> objOrder, 
					std::vector< std::vector< std::pair <Eigen::Isometry3d, float> > > unconditionedHypothesis,
					std::string scenePath, Eigen::Matrix4f camPose, cv::Mat depthImage, std::vector<float> cutOffScore){
		rootState = new uct_state::UCTState(0, unconditionedHypothesis[1], NULL);
		rootState->updateStateId(0);

		this->objOrder = objOrder;
		this->unconditionedHypothesis = unconditionedHypothesis;
		this->scenePath = scenePath;
		this->camPose = camPose;
		this->depthImage = depthImage;
		this->cutOffScore = cutOffScore;

		// initialize best state
		bestState = rootState;
		bestScore = 0;

		// initialize physics engine
		pSim = new physim::PhySim();
		pSim->addTable(tableHeight);
		for(int ii=0; ii<objOrder.size(); ii++)
			pSim->initRigidBody(objOrder[ii]->objName);

		search_begin_time = clock();
	}

	/********************************* function: UCTSearch::backupReward ***********************************
	*******************************************************************************************************/

	void UCTSearch::backupReward(uct_state::UCTState *selState, float reward){
		while(selState){
			selState->numExpansions = selState->numExpansions + 1;
			selState->hval = selState->hval + reward;
			selState = selState->parentState;
		}
	}

	/********************************* function: UCTSearch::defaultPolicy **********************************
	*******************************************************************************************************/

	float UCTSearch::defaultPolicy(uct_state::UCTState *selState){
		unsigned int maxDepth = objOrder.size();
		uct_state::UCTState* tmpState = new uct_state::UCTState(selState->numObjects, unconditionedHypothesis[selState->numObjects], selState->parentState);

		while(tmpState->numObjects < maxDepth){
			tmpState->numObjects++;
			int randHypothesis = rand() % unconditionedHypothesis[tmpState->numObjects].size();
			tmpState->updateNewObject(objOrder[tmpState->numObjects], unconditionedHypothesis[tmpState->numObjects][randHypothesis], maxDepth);
			tmpState->performTrICP(scenePath, 0.9);
			tmpState->correctPhysics(pSim, camPose, scenePath);
		}

		cv::Mat depth_image_defPolicy;
		tmpState->render(camPose, scenePath, depth_image_defPolicy);
		tmpState->computeCost(depth_image_defPolicy, depthImage);

		return tmpState->score;
	}

	/********************************* function: expand ****************************************************
	*******************************************************************************************************/

	uct_state::UCTState* UCTSearch::expand(uct_state::UCTState *currState){
		unsigned int maxDepth = objOrder.size();
		for(int ii=0; ii<currState->numChildren; ii++){
			if(currState->isExpanded[ii] == 0){
				uct_state::UCTState* childState = new uct_state::UCTState(currState->numObjects+1, unconditionedHypothesis[currState->numObjects+1], currState);
				childState->copyParent(currState);
				childState->updateStateId(ii);
				childState->updateNewObject(objOrder[currState->numObjects], unconditionedHypothesis[currState->numObjects][ii], maxDepth);
				currState->children.push_back(childState);
				return childState;
			}
		}

		std::cout << "UCTState::expand :: should never come here !!!" << std::endl;
		exit(-1);
	}

	/******************************** function: treePolicy **************************************************
	/*******************************************************************************************************/

	uct_state::UCTState* UCTSearch::treePolicy(uct_state::UCTState *currState){
		unsigned int maxDepth = objOrder.size();

		while(currState->numObjects < maxDepth){
			if(!currState->isFullyExpanded())
				return expand(currState);
			else
				currState = currState->getBestChild();
		}
		return currState;
	}

	/********************************* function: UCTSearch::performSearch ***********************************
	/*******************************************************************************************************/

	void UCTSearch::performSearch(){
		const clock_t begin_time = clock();
		
		numExpansions = 0;
		numRenders = 0;
		expansionTime = 0;
		renderTime = 0;

		while(1){

			if((float( clock () - begin_time ) /  CLOCKS_PER_SEC) > 600)
				break;

			uct_state::UCTState *selState = treePolicy(rootState);
			float reward = defaultPolicy(selState);
			backupReward(selState, reward);
		}

		std::cout<<"total number of state expansions: " << numExpansions <<std::endl;
		std::cout<<"mean expansion time: " << float(expansionTime/numExpansions) <<std::endl;
		std::cout<<"mean render time: " << float(renderTime/numRenders) <<std::endl;
	}

	/********************************* end of functions ****************************************************
	*******************************************************************************************************/
}