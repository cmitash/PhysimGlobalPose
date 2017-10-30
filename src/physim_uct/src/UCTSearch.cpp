#include <UCTSearch.hpp>

bool operator<(const uct_state::UCTState& lhs, const uct_state::UCTState& rhs) {
	return lhs.qval < rhs.qval;
}

namespace uct_search{
	clock_t search_begin_time;
	float tableHeight = 0.53;
	float trimICPthreshold = 0.9;
	int maxSearchTime = 120;
	int maxSearchIters = 300;
	int numExpansionsSearch;
	
	/********************************* function: constructor ***********************************************
	*******************************************************************************************************/

	UCTSearch::UCTSearch(std::vector<apc_objects::APCObjects*> objOrder, 
					std::vector< std::vector< std::pair <Eigen::Isometry3d, float> > > unconditionedHypothesis,
					std::string scenePath, Eigen::Matrix4f camPose, cv::Mat depthImage, std::vector<float> cutOffScore, int rootId){

		// initialize the root state
		int numChildNodesRoot = unconditionedHypothesis[0].size();
		rootState = new uct_state::UCTState(0, numChildNodesRoot, NULL);
		allStatePtrs.push_back(rootState);
		rootState->updateStateId(rootId);
		rootState->updateChildHval(unconditionedHypothesis[0]);
		
		// get scene information
		this->objOrder = objOrder;
		this->unconditionedHypothesis = unconditionedHypothesis;
		this->scenePath = scenePath;
		this->camPose = camPose;
		this->depthImage = depthImage;
		this->cutOffScore = cutOffScore;

		// initialize best state
		bestState = rootState;
		bestRenderScore = 0;

		// initialize physics engine
		pSim = new physim::PhySim();
		pSim->addTable(tableHeight);
		for(int ii=0; ii<objOrder.size(); ii++)
			pSim->initRigidBody(objOrder[ii]->objName);

		search_begin_time = clock();
		numExpansionsSearch = 0;
	}

	/********************************* function: destructor ************************************************
	*******************************************************************************************************/

	UCTSearch::~UCTSearch(){
		for(int ii=0; ii<allStatePtrs.size(); ii++){
			delete allStatePtrs[ii];
		}
	}

	/********************************* function: UCTSearch::backupReward ***********************************
	*******************************************************************************************************/

	void UCTSearch::backupReward(uct_state::UCTState *selState, float reward){
		while(selState){
			selState->numExpansions = selState->numExpansions + 1;
			selState->qval = selState->qval + reward;
			selState = selState->parentState;
		}
	}

	/********************************* function: UCTSearch::LCPPolicy **************************************
	*******************************************************************************************************/

	float UCTSearch::LCPPolicy(uct_state::UCTState *selState){
		unsigned int maxDepth = objOrder.size();

		uct_state::UCTState* tmpState = new uct_state::UCTState(selState->numObjects, selState->numChildren, selState->parentState);
		allStatePtrs.push_back(tmpState);
		tmpState->copyParent(selState);
		
		while(tmpState->numObjects < maxDepth){
			tmpState->numObjects++;

			int bestIdx = -1;
			float bestScoreLCP = 0;
			for(int ii=0; ii< unconditionedHypothesis[tmpState->numObjects-1].size(); ii++){
				if(unconditionedHypothesis[tmpState->numObjects-1][ii].second > bestScoreLCP){
					bestScoreLCP = unconditionedHypothesis[tmpState->numObjects-1][ii].second;
					bestIdx = ii;
				}
			}
			ofstream pFile;
		    pFile.open ((scenePath + "debug_search/debug.txt").c_str(), std::ofstream::out | std::ofstream::app);
			pFile << "Policy : " << numExpansionsSearch << " " << selState->stateId << " " << bestScoreLCP<< std::endl;
			pFile.close();

			tmpState->updateStateId(bestIdx);
			tmpState->updateNewObject(objOrder[tmpState->numObjects-1], unconditionedHypothesis[tmpState->numObjects-1][bestIdx], maxDepth);
			tmpState->performTrICP(scenePath, 0.9);
			tmpState->correctPhysics(pSim, camPose, scenePath);
			tmpState->render(camPose, scenePath);
			numExpansionsSearch++;
		}

		tmpState->computeCost(depthImage);
		if(tmpState->renderScore > bestRenderScore){
			bestState = tmpState;
			bestRenderScore = tmpState->renderScore;

			#ifdef DBG_SUPER4PCS
			for(int ii=0; ii<objOrder.size();ii++){
		      Eigen::Matrix4f tform;
		      utilities::convertToMatrix(bestState->objects[ii].second, tform);
		      utilities::convertToWorld(tform, camPose);
		      utilities::writePoseToFile(tform, bestState->objects[ii].first->objName, scenePath, "debug_search/after_search");

		      ofstream pFile;
		      pFile.open ((scenePath + "debug_search/times_" + bestState->objects[ii].first->objName + ".txt").c_str(), std::ofstream::out | std::ofstream::app);
			  pFile << numExpansionsSearch << " " << bestRenderScore << " " << tmpState->stateId  << std::endl;
			  pFile.close();
		    } 
			#endif
		}

		return tmpState->renderScore;
	}

	/********************************* function: UCTSearch::defaultPolicy **********************************
	*******************************************************************************************************/

	float UCTSearch::defaultPolicy(uct_state::UCTState *selState){
		unsigned int maxDepth = objOrder.size();

		uct_state::UCTState* tmpState = new uct_state::UCTState(selState->numObjects, selState->numChildren, selState->parentState);
		allStatePtrs.push_back(tmpState);
		tmpState->copyParent(selState);
		
		while(tmpState->numObjects < maxDepth){
			tmpState->numObjects++;

			ofstream pFile;
		    pFile.open ((scenePath + "debug_search/debug.txt").c_str(), std::ofstream::out | std::ofstream::app);
			pFile << "Policy : " << numExpansionsSearch << " " << selState->stateId << std::endl;
			pFile.close();

			// random policy
			int randHypothesis = rand() % unconditionedHypothesis[tmpState->numObjects-1].size();
			tmpState->updateStateId(randHypothesis);
			tmpState->updateNewObject(objOrder[tmpState->numObjects-1], unconditionedHypothesis[tmpState->numObjects-1][randHypothesis], maxDepth);
			tmpState->performTrICP(scenePath, 0.9);
			tmpState->correctPhysics(pSim, camPose, scenePath);
			tmpState->render(camPose, scenePath);
			numExpansionsSearch++;
		}

		std::cout << "UCTSearch::defaultPolicy:: Rendered Stateid: " << tmpState->stateId << std::endl;
		
		tmpState->computeCost(depthImage);
		if(tmpState->renderScore > bestRenderScore){
			bestState = tmpState;
			bestRenderScore = tmpState->renderScore;

			#ifdef DBG_SUPER4PCS
			for(int ii=0; ii<objOrder.size();ii++){
		      Eigen::Matrix4f tform;
		      utilities::convertToMatrix(bestState->objects[ii].second, tform);
		      utilities::convertToWorld(tform, camPose);
		      utilities::writePoseToFile(tform, bestState->objects[ii].first->objName, scenePath, "debug_search/after_search");

		      ofstream pFile;
		      pFile.open ((scenePath + "debug_search/times_" + bestState->objects[ii].first->objName + ".txt").c_str(), std::ofstream::out | std::ofstream::app);
			  pFile << numExpansionsSearch << " " << bestRenderScore << " " << tmpState->stateId  << std::endl;
			  pFile.close();
		    } 
			#endif
		}

		return tmpState->renderScore;
	}

	/********************************* function: expand ****************************************************
	this function is called when the current state is not fully expanded.
	*******************************************************************************************************/

	uct_state::UCTState* UCTSearch::expand(uct_state::UCTState *currState){
		unsigned int maxDepth = objOrder.size();

		// find a non-expanded node with the best heuristic value.
		int bestChildIdx = -1;
		float bestHval = 0;
		for(int ii=0; ii<currState->numChildren; ii++){
			if(currState->isExpanded[ii] == 0 && currState->hval[ii] > bestHval){
				bestHval = currState->hval[ii];
				bestChildIdx = ii;
			}
		}

		ofstream pFile;
	    pFile.open ((scenePath + "debug_search/debug.txt").c_str(), std::ofstream::out | std::ofstream::app);
		pFile << numExpansionsSearch << " " << currState->stateId << " " << bestHval<< std::endl;
		pFile.close();

		int numObjectsChildNode = currState->numObjects + 1;

		int numChildNodesForChildNode = 0;
		if(numObjectsChildNode < maxDepth)
			numChildNodesForChildNode = unconditionedHypothesis[numObjectsChildNode].size();

		uct_state::UCTState* childState = new uct_state::UCTState(numObjectsChildNode, numChildNodesForChildNode, currState);
		allStatePtrs.push_back(childState);
		childState->copyParent(currState);
		childState->updateStateId(bestChildIdx);
		childState->updateNewObject(objOrder[currState->numObjects], unconditionedHypothesis[currState->numObjects][bestChildIdx], maxDepth);
		childState->updateChildHval(unconditionedHypothesis[currState->numObjects]);
		childState->performTrICP(scenePath, 0.9);
		childState->correctPhysics(pSim, camPose, scenePath);
		childState->render(camPose, scenePath);

		currState->children.push_back(childState);
		currState->isExpanded[bestChildIdx] = 1;

		numExpansionsSearch++;
		return childState;
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

			// if((float( clock () - begin_time ) /  CLOCKS_PER_SEC) > maxSearchTime)
			// 	break;

			if(numExpansionsSearch > maxSearchIters)
				break;
			
			std::cout << "*****UCTSearch::performSearch::Begin()*****" << std::endl;
			uct_state::UCTState *selState = treePolicy(rootState);
			float reward = defaultPolicy(selState);
			backupReward(selState, reward);
			std::cout << "*****UCTSearch::performSearch::End()*****" << std::endl;
			std::cout << std::endl;
		}
	}

	/********************************* end of functions ****************************************************
	*******************************************************************************************************/
}