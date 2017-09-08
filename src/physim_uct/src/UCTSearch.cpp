#include <UCTSearch.hpp>

bool operator<(const uct_state::UCTState& lhs, const uct_state::UCTState& rhs) {
	return lhs.hval < rhs.hval;
}

namespace uct_search{
	clock_t search_begin_time;
	float tableHeight = 0.545;
	float trimICPthreshold = 0.9;
	int maxSearchTime = 60;
	
	/********************************* function: constructor ***********************************************
	*******************************************************************************************************/

	UCTSearch::UCTSearch(std::vector<apc_objects::APCObjects*> objOrder, 
					std::vector< std::vector< std::pair <Eigen::Isometry3d, float> > > unconditionedHypothesis,
					std::string scenePath, Eigen::Matrix4f camPose, cv::Mat depthImage, std::vector<float> cutOffScore){

		int numChildNodesRoot = unconditionedHypothesis[0].size();
		rootState = new uct_state::UCTState(0, numChildNodesRoot, NULL);
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

		uct_state::UCTState* tmpState = new uct_state::UCTState(selState->numObjects, selState->numChildren, selState->parentState);
		tmpState->copyParent(selState);
		
		while(tmpState->numObjects < maxDepth){
			tmpState->numObjects++;
			int randHypothesis = rand() % unconditionedHypothesis[tmpState->numObjects-1].size();

			tmpState->updateStateId(randHypothesis);
			tmpState->updateNewObject(objOrder[tmpState->numObjects-1], unconditionedHypothesis[tmpState->numObjects-1][randHypothesis], maxDepth);
			tmpState->performTrICP(scenePath, 0.9);
			tmpState->correctPhysics(pSim, camPose, scenePath);
		}

		std::cout << "UCTSearch::defaultPolicy:: Rendered Stateid: " << tmpState->stateId << std::endl;

		cv::Mat depth_image_defPolicy;
		tmpState->render(camPose, scenePath, depth_image_defPolicy);
		tmpState->computeCost(depth_image_defPolicy, depthImage);

		if(tmpState->score > bestScore){
			bestState = tmpState;
			bestScore = tmpState->score;

			#ifdef DBG_SUPER4PCS
			for(int ii=0; ii<objOrder.size();ii++){
		      Eigen::Matrix4f tform;
		      utilities::convertToMatrix(bestState->objects[ii].second, tform);
		      utilities::convertToWorld(tform, camPose);
		      utilities::writePoseToFile(tform, bestState->objects[ii].first->objName, scenePath, "debug_search/after_search");

		      ofstream pFile;
		      pFile.open ((scenePath + "debug_search/times_" + bestState->objects[ii].first->objName + ".txt").c_str(), std::ofstream::out | std::ofstream::app);
			  pFile << (float( clock () - search_begin_time ) /  CLOCKS_PER_SEC) << std::endl;
			  pFile.close();
		    } 
			#endif
		}

		return tmpState->score;
	}

	/********************************* function: expand ****************************************************
	*******************************************************************************************************/

	uct_state::UCTState* UCTSearch::expand(uct_state::UCTState *currState){
		std::cout << "UCTSearch::expand:: stateid: " << currState->stateId << std::endl;

		unsigned int maxDepth = objOrder.size();

		for(int ii=0; ii<currState->numChildren; ii++){
			if(currState->isExpanded[ii] == 0){
				currState->isExpanded[ii] = 1;

				int numChildNodesForChildNode = 0;
				if((currState->numObjects + 1) < maxDepth)
					numChildNodesForChildNode = unconditionedHypothesis[currState->numObjects+1].size();

				uct_state::UCTState* childState = new uct_state::UCTState(currState->numObjects+1, numChildNodesForChildNode, currState);
				childState->copyParent(currState);
				childState->updateStateId(ii);
				childState->updateNewObject(objOrder[currState->numObjects], unconditionedHypothesis[currState->numObjects][ii], maxDepth);
				childState->performTrICP(scenePath, 0.9);
				childState->correctPhysics(pSim, camPose, scenePath);
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

		std::cout << "UCTSearch::treePolicy:: stateid: " << currState->stateId << std::endl;
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
		
		while(1){

			if((float( clock () - begin_time ) /  CLOCKS_PER_SEC) > maxSearchTime)
				break;

			std::cout << "*****UCTSearch::performSearch::Begin()*****" << std::endl;
			uct_state::UCTState *selState = treePolicy(rootState);
			float reward = defaultPolicy(selState);
			backupReward(selState, reward);
			std::cout << "*****UCTSearch::performSearch::End()*****" << std::endl;
			std::cout << std::endl;
		}

		// unsigned int maxDepth = objOrder.size();
		// while(bestState->numObjects < maxDepth){
		// 	bestState = bestState->getBestChild();
		// 	std::cout << "UCTSearch::performSearch:: stateId: " << bestState->stateId << std::endl;
		// }
	}

	/********************************* end of functions ****************************************************
	*******************************************************************************************************/
}