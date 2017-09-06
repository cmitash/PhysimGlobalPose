#include <Search.hpp>

bool operator<(const state::State& lhs, const state::State& rhs) {
	return lhs.hval < rhs.hval;
}

namespace search{

	int numExpansions;
	int numRenders;
	float expansionTime;
	float renderTime;
	clock_t search_begin_time;
	float tableHeight = 0.545;
	float trimICPthreshold = 0.9;
	
	/********************************* function: constructor ***********************************************
	*******************************************************************************************************/

	Search::Search(std::vector<apc_objects::APCObjects*> objOrder, 
					std::vector< std::vector< std::pair <Eigen::Isometry3d, float> > > unconditionedHypothesis,
					std::string scenePath, Eigen::Matrix4f camPose, cv::Mat depthImage, std::vector<float> cutOffScore){
		rootState = new state::State(0);
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

	/********************************* function: expandNode ***********************************************
	*******************************************************************************************************/

	void Search::expandNode(state::State* expState){
		expState->expand();

		const clock_t exp_begin_time = clock();

		expState->performTrICP(scenePath, trimICPthreshold);
		expState->correctPhysics(pSim, camPose, scenePath);
		expState->performTrICP(scenePath, trimICPthreshold/2);

		expansionTime += (float( clock () - exp_begin_time ) /  CLOCKS_PER_SEC);

		unsigned int maxDepth = objOrder.size();
		if(expState->numObjects == maxDepth){
			cv::Mat depth_image;

			const clock_t render_begin_time = clock();
			expState->render(camPose, scenePath, depth_image);
			numRenders++;
			expState->computeCost(depth_image, depthImage);

			renderTime += (float( clock () - render_begin_time ) /  CLOCKS_PER_SEC);

			if(expState->score > bestScore){
				bestState = expState;
				bestScore = expState->score;

			#ifdef DBG_SUPER4PCS
				for(int ii=0; ii<objOrder.size();ii++){
			      Eigen::Matrix4f tform;
			      utilities::convertToMatrix(bestState->objects[ii].second, tform);
			      utilities::convertToWorld(tform, camPose);
			      utilities::writePoseToFile(tform, bestState->objects[ii].first->objName, scenePath, "after_search");

			      ofstream pFile;
			      pFile.open ((scenePath + "debug/times_" + bestState->objects[ii].first->objName + ".txt").c_str(), std::ofstream::out | std::ofstream::app);
				  pFile << (float( clock () - search_begin_time ) /  CLOCKS_PER_SEC) << std::endl;
				  pFile.close();
			    }
			    
			#endif

			}
		}
		else{
			unsigned int nextDepthLevel = expState->numObjects + 1;
			for(int ii = 0; ii < unconditionedHypothesis[nextDepthLevel - 1].size(); ii++){
				if(unconditionedHypothesis[nextDepthLevel - 1][ii].second >= cutOffScore[nextDepthLevel - 1]){
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
			
			if((float( clock () - begin_time ) /  CLOCKS_PER_SEC) > 600000)
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