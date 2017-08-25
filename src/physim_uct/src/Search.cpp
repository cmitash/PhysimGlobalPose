#include <Search.hpp>

bool operator<(const state::State& lhs, const state::State& rhs) {
	return lhs.hval < rhs.hval;
}

int numExpansions;
int numRenders;
float expansionTime;
float renderTime;
float thresholdLCP = 0;

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

		#ifdef DBG_SUPER4PCS
			float avgRotErr, avgTransErr;
			for(int i = 0;i < expState->objects.size();i++){
				Eigen::Matrix4f tform;
				utilities::convertToMatrix(expState->objects[i].second, tform);
				utilities::convertToWorld(tform, currScene->camPose);

				ifstream gtPoseFile;
				Eigen::Matrix4f gtPose;
				gtPose.setIdentity();
				gtPoseFile.open((currScene->scenePath + "gt_pose_" + expState->objects[i].first->objName + ".txt").c_str(), std::ifstream::in);
				gtPoseFile >> gtPose(0,0) >> gtPose(0,1) >> gtPose(0,2) >> gtPose(0,3) 
					>> gtPose(1,0) >> gtPose(1,1) >> gtPose(1,2) >> gtPose(1,3)
					>> gtPose(2,0) >> gtPose(2,1) >> gtPose(2,2) >> gtPose(2,3);
				gtPoseFile.close();

				float rotErr, transErr;
				utilities::getPoseError(tform, gtPose, expState->objects[i].first->symInfo, rotErr, transErr);
				avgRotErr += rotErr;
				avgTransErr += transErr;
			}
			avgRotErr /= expState->objects.size();
			avgTransErr /= expState->objects.size();
			ofstream scoreFile;
			scoreFile.open ((currScene->scenePath + "debug/scores.txt").c_str(), std::ofstream::out | std::ofstream::app);
			scoreFile << avgRotErr << " " << avgTransErr << " " << expState->score << std::endl;
			scoreFile.close();
		#endif

			renderTime += (float( clock () - render_begin_time ) /  CLOCKS_PER_SEC);

			if(expState->score < bestScore){
				bestState = expState;
				bestScore = expState->score;
			}
		}
		else{
			unsigned int nextDepthLevel = expState->numObjects + 1;
			for(int ii = 0; ii < unconditionedHypothesis[nextDepthLevel - 1].size(); ii++){
				if(unconditionedHypothesis[nextDepthLevel - 1][ii].second >= thresholdLCP*currScene->max4PCSPose[nextDepthLevel - 1].second){
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

	/********************************* function: UCTSearch *************************************************
	*******************************************************************************************************/

	// void Search::expand(){
		
	// }

	// /********************************* function: treePolicy ************************************************
	// *******************************************************************************************************/

	// void Search::treePolicy(state::State *currState){
	// 	while()
	// }

	// /********************************* function: UCTSearch *************************************************
	// *******************************************************************************************************/

	// void Search::UCTSearch(){
	// 	const clock_t begin_time = clock();
		
	// 	numExpansions = 0;
	// 	numRenders = 0;
	// 	expansionTime = 0;
	// 	renderTime = 0;

	// 	while(1){

	// 		if((float( clock () - begin_time ) /  CLOCKS_PER_SEC) > 600)
	// 			break;

	// 		state::State *selState = treePolicy(rootState);
	// 		float reward = defaultPolicy(selState);
	// 		backupReward(selState, reward);
	// 	}

	// 	std::cout<<"total number of state expansions: " << numExpansions <<std::endl;
	// 	std::cout<<"mean expansion time: " << float(expansionTime/numExpansions) <<std::endl;
	// 	std::cout<<"mean render time: " << float(renderTime/numRenders) <<std::endl;
	// }

	/********************************* end of functions ****************************************************
	*******************************************************************************************************/
}