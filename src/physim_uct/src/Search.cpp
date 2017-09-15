#include <Search.hpp>

namespace search{

	int numExpansionsSearch;
	float tableHeight = 0.545;
	float trimICPthreshold = 0.9;
	
	/********************************* function: constructor ***********************************************
	*******************************************************************************************************/

	Search::Search(std::vector<apc_objects::APCObjects*> objOrder, 
					std::vector< std::vector< std::pair <Eigen::Isometry3d, float> > > unconditionedHypothesis,
					std::string scenePath, Eigen::Matrix4f camPose, cv::Mat obsDepthImage, std::vector<float> cutOffScore, int rootId){
		rootState = new state::State(0);
		rootState->updateStateId(rootId);

		this->objOrder = objOrder;
		this->unconditionedHypothesis = unconditionedHypothesis;
		this->scenePath = scenePath;
		this->camPose = camPose;
		this->obsDepthImage = obsDepthImage;
		this->cutOffScore = cutOffScore;

		// initialize best state
		bestState = rootState;
		bestScore = 0;

		// initialize physics engine
		pSim = new physim::PhySim();
		pSim->addTable(tableHeight);
		for(int ii=0; ii<objOrder.size(); ii++)
			pSim->initRigidBody(objOrder[ii]->objName);

		numExpansionsSearch = 0;
	}

	/********************************* function: expandNode ***********************************************
	*******************************************************************************************************/

	void Search::expandNode(state::State* expState){
		expState->expand();

		expState->performTrICP(scenePath, trimICPthreshold);
		expState->correctPhysics(pSim, camPose, scenePath);
		expState->render(camPose, scenePath);

		ofstream pFile;
	    pFile.open ((scenePath + "debug_search/debug.txt").c_str(), std::ofstream::out | std::ofstream::app);
		pFile << numExpansionsSearch << " " << expState->stateId << " " << expState->score << " " << expState->hval << std::endl;
		pFile.close();

		unsigned int maxDepth = objOrder.size();
		if(expState->numObjects == maxDepth){
			expState->computeCost(obsDepthImage);

			if(expState->score > bestScore){
				bestState = expState;
				bestScore = expState->score;

			#ifdef DBG_SUPER4PCS
				for(int ii=0; ii<objOrder.size();ii++){
			      Eigen::Matrix4f tform;
			      utilities::convertToMatrix(bestState->objects[ii].second, tform);
			      utilities::convertToWorld(tform, camPose);
			      utilities::writePoseToFile(tform, bestState->objects[ii].first->objName, scenePath, "debug_search/after_search");

			      ofstream pFile;
			      pFile.open ((scenePath + "debug_search/times_" + bestState->objects[ii].first->objName + ".txt").c_str(), std::ofstream::out | std::ofstream::app);
				  pFile << numExpansionsSearch << " " << bestScore << " " << expState->stateId  << std::endl;
				  pFile.close();
			    }
			#endif
			}
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
			
			if((float( clock () - begin_time ) /  CLOCKS_PER_SEC) > 600000)
				break;
			
			state::State *expState = pq.top();
			pq.pop();
			numExpansionsSearch++;
			expandNode(expState);
		}
	}

	/********************************* end of functions ****************************************************
	*******************************************************************************************************/
}