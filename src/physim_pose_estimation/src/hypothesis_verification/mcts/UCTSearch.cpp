#include <UCTSearch.hpp>

bool operator<(const uct_state::UCTState& lhs, const uct_state::UCTState& rhs) {
	return lhs.qval < rhs.qval;
}

namespace uct_search{
	clock_t search_begin_time;
	float trimICPthreshold = 1.0;
	int maxSearchTime = 60;
	int numExpansionsSearch;
	
	/********************************* function: constructor ***********************************************
	*******************************************************************************************************/

	UCTSearch::UCTSearch(std::vector<apc_objects::APCObjects*> objOrder, std::vector<float> tableParams,
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
		bestState = new uct_state::UCTState(0, numChildNodesRoot, NULL);
		bestRenderScore = INT_MAX;

		// initialize physics engine
		pSim = new physim::PhySim(tableParams);
		pSim->addTable(tableParams);
		for(int ii=0; ii<objOrder.size(); ii++)
			pSim->initRigidBody(objOrder[ii]->objName);

		search_begin_time = clock();
		numExpansionsSearch = 0;
	}

	/********************************* function: destructor ************************************************
	*******************************************************************************************************/

	UCTSearch::~UCTSearch(){
		std::cout << "UCTSearch::~UCTSearch()::Total number of State pointers: " << allStatePtrs.size() << std::endl;
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

		// If the selected state is a leaf, it should just return it's render score
		if(selState->numObjects == maxDepth)
			return selState->renderScore;

		uct_state::UCTState* tmpState = new uct_state::UCTState(selState->numObjects, selState->numChildren, selState->parentState);
		tmpState->copyParent(selState);
		
		// use LCP score to chose objects until we reach the leaf node
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

			tmpState->updateStateId(bestIdx);
			tmpState->updateNewObject(objOrder[tmpState->numObjects-1], unconditionedHypothesis[tmpState->numObjects-1][bestIdx], maxDepth);
			tmpState->performTrICP(scenePath, trimICPthreshold);
			tmpState->correctPhysics(pSim, camPose, scenePath);
			tmpState->render(camPose, scenePath);
		}

		tmpState->computeCost(depthImage);
		unsigned int currScore = tmpState->renderScore;

		ofstream pFile;
	    pFile.open ((scenePath + "debug_search/debug.txt").c_str(), std::ofstream::out | std::ofstream::app);
		pFile << "UCTSearch::LCPPolicy:: renderedState: " << tmpState->stateId << ", renderScore: " << currScore << std::endl;
		pFile.close();

		if(currScore < bestRenderScore){
			bestState->numObjects = tmpState->numObjects;
			bestState->objects = std::vector<std::pair<apc_objects::APCObjects*, Eigen::Isometry3d> >(tmpState->numObjects);
			for(int jj=0;jj<tmpState->numObjects;jj++)
      			bestState->objects[jj] = tmpState->objects[jj];

			bestRenderScore = currScore;

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
		}

		delete tmpState;

		return currScore;
	}

	/********************************* function: UCTSearch::defaultPolicy **********************************
	*******************************************************************************************************/

	float UCTSearch::defaultPolicy(uct_state::UCTState *selState){
		unsigned int maxDepth = objOrder.size();

		// If the selected state is a leaf, it should just return it's render score
		if(selState->numObjects == maxDepth)
			return selState->renderScore;

		uct_state::UCTState* tmpState = new uct_state::UCTState(selState->numObjects, selState->numChildren, selState->parentState);
		tmpState->copyParent(selState);
		
		while(tmpState->numObjects < maxDepth){
			tmpState->numObjects++;

			// random policy
			int randHypothesis = rand() % unconditionedHypothesis[tmpState->numObjects-1].size();
			tmpState->updateStateId(randHypothesis);
			tmpState->updateNewObject(objOrder[tmpState->numObjects-1], unconditionedHypothesis[tmpState->numObjects-1][randHypothesis], maxDepth);
			tmpState->performTrICP(scenePath, trimICPthreshold);
			tmpState->correctPhysics(pSim, camPose, scenePath);
			tmpState->render(camPose, scenePath);
		}

		tmpState->computeCost(depthImage);
		unsigned int currScore = tmpState->renderScore;

		ofstream pFile;
	    pFile.open ((scenePath + "debug_search/debug.txt").c_str(), std::ofstream::out | std::ofstream::app);
		pFile << "UCTSearch::defaultPolicy:: randomState: " << tmpState->stateId << ", renderScore: " << currScore << std::endl;
		pFile.close();

		if(currScore < bestRenderScore){
			bestState->numObjects = tmpState->numObjects;
			bestState->objects = std::vector<std::pair<apc_objects::APCObjects*, Eigen::Isometry3d> >(tmpState->numObjects);
			for(int jj=0;jj<tmpState->numObjects;jj++)
      			bestState->objects[jj] = tmpState->objects[jj];

			bestRenderScore = currScore;

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
		}

		delete tmpState;

		return currScore;
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
		childState->performTrICP(scenePath, trimICPthreshold);
		childState->correctPhysics(pSim, camPose, scenePath);
		childState->render(camPose, scenePath);
		childState->computeCost(depthImage);

		// if the expanded node is the leaf node
		if(childState->numObjects == maxDepth && childState->renderScore < bestRenderScore){
			bestState->numObjects = childState->numObjects;
			bestState->objects = std::vector<std::pair<apc_objects::APCObjects*, Eigen::Isometry3d> >(childState->numObjects);
			for(int jj=0;jj<childState->numObjects;jj++)
      			bestState->objects[jj] = childState->objects[jj];

			bestRenderScore = childState->renderScore;

			for(int ii=0; ii<objOrder.size();ii++){
		      Eigen::Matrix4f tform;
		      utilities::convertToMatrix(bestState->objects[ii].second, tform);
		      utilities::convertToWorld(tform, camPose);
		      utilities::writePoseToFile(tform, bestState->objects[ii].first->objName, scenePath, "debug_search/after_search");

		      ofstream pFile;
		      pFile.open ((scenePath + "debug_search/times_" + bestState->objects[ii].first->objName + ".txt").c_str(), std::ofstream::out | std::ofstream::app);
			  pFile << numExpansionsSearch << " " << bestRenderScore << " " << childState->stateId  << std::endl;
			  pFile.close();
		    } 
		}

		currState->children.push_back(childState);
		currState->isExpanded[bestChildIdx] = 1;

		numExpansionsSearch++;

		// write into the debug file
		ofstream pFile;
	    pFile.open ((scenePath + "debug_search/debug.txt").c_str(), std::ofstream::out | std::ofstream::app);
		pFile << "UCTSearch::expand:: numExpansionsSearch: " << numExpansionsSearch << 
					", currState: " << currState->stateId << ", childState: " << childState->stateId <<
					", bestHval: " << bestHval<< ", renderScore: " << childState->renderScore <<std::endl;
		pFile.close();

		return childState;
	}

	/******************************** function: treePolicy **************************************************
	/*******************************************************************************************************/

	uct_state::UCTState* UCTSearch::treePolicy(uct_state::UCTState *currState){
		unsigned int maxDepth = objOrder.size();

		while(currState->numObjects < maxDepth){
			if(!currState->isFullyExpanded())
				return expand(currState);
			else
				currState = currState->getBestChild(scenePath);
		}
		return currState;
	}

	/********************************* function: UCTSearch::performSearch ***********************************
	/*******************************************************************************************************/

	void UCTSearch::performSearch(){
		const clock_t begin_time = clock();
		
		int numObjects = objOrder.size();

		int stoppingCriteria = 0;
		for (int ii=0; ii<=numObjects; ii++)
			stoppingCriteria += pow(25, ii);

		while(1){

			// stopping criterias
			if(numExpansionsSearch >= stoppingCriteria || 
					(float( clock () - begin_time ) /  CLOCKS_PER_SEC) > maxSearchTime)
				break;
			
			std::cout << "UCTSearch::performSearch:: Number of states expanded: " << numExpansionsSearch << std::endl;
			uct_state::UCTState *selState = treePolicy(rootState);
			float reward = LCPPolicy(selState);
			backupReward(selState, reward);
		}
	}

	/********************************* end of functions ****************************************************
	*******************************************************************************************************/
}