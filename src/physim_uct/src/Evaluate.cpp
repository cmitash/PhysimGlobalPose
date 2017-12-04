#include <Evaluate.hpp>

std::vector< std::pair <apc_objects::APCObjects*, Eigen::Matrix4f> > groundTruth;
int evalEMD = 0;

#define DATASET_SIZE 126

namespace evaluate{

	Evaluate::Evaluate(): 
	rotErr_super4pcs(DATASET_SIZE, INT_MAX),
	transErr_super4pcs(DATASET_SIZE, INT_MAX),
	emdErr_super4pcs(DATASET_SIZE, INT_MAX),

	rotErr_super4pcsICP(DATASET_SIZE, INT_MAX),
	transErr_super4pcsICP(DATASET_SIZE, INT_MAX),
	emdErr_super4pcsICP(DATASET_SIZE, INT_MAX),

	rotErr_allhypothesisMinRot(DATASET_SIZE, INT_MAX),
	transErr_allhypothesisMinRot(DATASET_SIZE, INT_MAX),
	emdErr_allhypothesisMinRot(DATASET_SIZE, INT_MAX),

	rotErr_allhypothesisMintrans(DATASET_SIZE, INT_MAX),
	transErr_allhypothesisMintrans(DATASET_SIZE, INT_MAX),
	emdErr_allhypothesisMintrans(DATASET_SIZE, INT_MAX),

	rotErr_clusterhypothesisMinRot(DATASET_SIZE, INT_MAX),
	transErr_clusterhypothesisMinRot(DATASET_SIZE, INT_MAX),
	emdErr_clusterhypothesisMinRot(DATASET_SIZE, INT_MAX),

	rotErr_clusterhypothesisMintrans(DATASET_SIZE, INT_MAX),
	transErr_clusterhypothesisMintrans(DATASET_SIZE, INT_MAX),
	emdErr_clusterhypothesisMintrans(DATASET_SIZE, INT_MAX),

	rotErr_clusterhypothesisMinEmd(DATASET_SIZE, INT_MAX),
	transErr_clusterhypothesisMinEmd(DATASET_SIZE, INT_MAX),
	emdErr_clusterhypothesisMinEmd(DATASET_SIZE, INT_MAX),

	rotErr_searchFinal(DATASET_SIZE), 
	transErr_searchFinal(DATASET_SIZE),
	emdErr_searchFinal(DATASET_SIZE),

	rotErr_PCA(DATASET_SIZE),
	transErr_PCA(DATASET_SIZE),
	emdErr_PCA(DATASET_SIZE){}

/********************************* function: getRangeEMDComputation ************************************
*******************************************************************************************************/

static void getRangeEMDComputation(PointCloud::Ptr pclSegment, Eigen::Matrix4f camPose, std::pair<float, float> &xrange, 
									std::pair<float, float> &yrange, std::pair<float, float> &zrange){
	std:: cout << "Evaluate::getRangeEMDComputation" << std::endl;

	float x_min = INT_MAX, x_max = -INT_MAX;
	float y_min = INT_MAX, y_max = -INT_MAX;
	float z_min = INT_MAX, z_max = -INT_MAX;

	PointCloud::Ptr pclSegmentWorld (new PointCloud);
	pcl::transformPointCloud(*pclSegment, *pclSegmentWorld, camPose);
	for(int ii=0; ii<pclSegmentWorld->points.size(); ii++){
	    if(pclSegmentWorld->points[ii].x < x_min)x_min = pclSegmentWorld->points[ii].x;
	    if(pclSegmentWorld->points[ii].y < y_min)y_min = pclSegmentWorld->points[ii].y;
	    if(pclSegmentWorld->points[ii].z < z_min)z_min = pclSegmentWorld->points[ii].z;

	    if(pclSegmentWorld->points[ii].x > x_max)x_max = pclSegmentWorld->points[ii].x;
	    if(pclSegmentWorld->points[ii].y > y_max)y_max = pclSegmentWorld->points[ii].y;
	    if(pclSegmentWorld->points[ii].z > z_max)z_max = pclSegmentWorld->points[ii].z;
	}

	xrange = std::make_pair(x_min-0.05, x_max+0.05);
	yrange = std::make_pair(y_min-0.05, y_max+0.05);
	zrange = std::make_pair(z_min-0.05, z_max+0.05);
}

/********************************* function: getSuper4pcsError *****************************************
*******************************************************************************************************/
void Evaluate::getSuper4pcsError(scene::Scene *currScene, int sceneIdx){
	std:: cout << "Evaluate::getSuper4pcsError" << std::endl;

	for(int obIdx=0;obIdx<currScene->sceneObjs.size();obIdx++){
		ifstream super4pcsFile;
		Eigen::Matrix4f testPose;
		testPose.setIdentity();
		super4pcsFile.open((currScene->scenePath + "debug_super4PCS/super4pcs_" + currScene->sceneObjs[obIdx]->objName + ".txt").c_str(), std::ifstream::in);

		float rotErr, transErr, emdErr=0;
		super4pcsFile >> testPose(0,0) >> testPose(0,1) >> testPose(0,2) >> testPose(0,3) 
					 >> testPose(1,0) >> testPose(1,1) >> testPose(1,2) >> testPose(1,3)
					 >> testPose(2,0) >> testPose(2,1) >> testPose(2,2) >> testPose(2,3);
		
		utilities::getPoseError(testPose, groundTruth[obIdx].second, groundTruth[obIdx].first->symInfo, rotErr, transErr);
	
		std::pair<float, float> xrange, yrange, zrange;
		if(evalEMD){
			getRangeEMDComputation(currScene->sceneObjs[obIdx]->pclSegment, currScene->camPose, xrange, yrange, zrange);
			utilities::getEMDError(testPose, groundTruth[obIdx].second, currScene->sceneObjs[obIdx]->pclModel, emdErr, xrange, yrange, zrange);
		}
		rotErr_super4pcs[sceneIdx*3 + obIdx] = rotErr;
		transErr_super4pcs[sceneIdx*3 + obIdx] = transErr;
		emdErr_super4pcs[sceneIdx*3 + obIdx] = emdErr;
		
		ofstream resFile;
		resFile.open ("/home/chaitanya/PoseDataset17/resultSuper4pcs.txt", std::ofstream::out|std::ofstream::app);
		resFile <<"getSuper4pcsError: " << sceneIdx*3 + obIdx << " " << rotErr << " " << transErr << " " << emdErr << std::endl;
		resFile.close();

		super4pcsFile >> testPose(0,0) >> testPose(0,1) >> testPose(0,2) >> testPose(0,3) 
					 >> testPose(1,0) >> testPose(1,1) >> testPose(1,2) >> testPose(1,3)
					 >> testPose(2,0) >> testPose(2,1) >> testPose(2,2) >> testPose(2,3);
			
		utilities::getPoseError(testPose, groundTruth[obIdx].second, groundTruth[obIdx].first->symInfo, rotErr, transErr);
		rotErr_super4pcsICP[sceneIdx*3 + obIdx] = rotErr;
		transErr_super4pcsICP[sceneIdx*3 + obIdx] = transErr;

		if(evalEMD){
			utilities::getEMDError(testPose, groundTruth[obIdx].second, currScene->sceneObjs[obIdx]->pclModel, emdErr, xrange, yrange, zrange);	
		}
		emdErr_super4pcsICP[sceneIdx*3 + obIdx] = emdErr;

		resFile.open ("/home/chaitanya/PoseDataset17/resultSuper4pcsICP.txt", std::ofstream::out| std::ofstream::app);
		resFile <<"getSuper4pcsError: " << sceneIdx*3 + obIdx << " " << rotErr << " " << transErr << " " << emdErr << std::endl;
		resFile.close();

		super4pcsFile.close();
	}
}

/********************************* function: getSearchResults *****************************************
*******************************************************************************************************/
void Evaluate::getSearchResults(scene::Scene *currScene, int sceneIdx){
	std::cout << "Evaluate::getSearchResults" << std::endl;

	for(int obIdx=0;obIdx<currScene->sceneObjs.size();obIdx++){
		ifstream super4pcsFile, timeFile;
		Eigen::Matrix4f testPose, finalState;
		float rotErr, transErr, emdErr=0;
		float time;
		unsigned int score;
		std::string bestState;
		std::pair<float, float> xrange, yrange, zrange;

		testPose.setIdentity();
		super4pcsFile.open((currScene->scenePath + "debug_search/after_search_" + currScene->sceneObjs[obIdx]->objName + ".txt").c_str(), std::ifstream::in);
		timeFile.open((currScene->scenePath + "debug_search/times_" + currScene->sceneObjs[obIdx]->objName + ".txt").c_str(), std::ifstream::in);
		
		while(super4pcsFile >> testPose(0,0) >> testPose(0,1) >> testPose(0,2) >> testPose(0,3) 
					 >> testPose(1,0) >> testPose(1,1) >> testPose(1,2) >> testPose(1,3)
					 >> testPose(2,0) >> testPose(2,1) >> testPose(2,2) >> testPose(2,3)){
			

			utilities::getPoseError(testPose, groundTruth[obIdx].second, groundTruth[obIdx].first->symInfo, rotErr, transErr);
			timeFile >> time >> score >> bestState;

			rotErr_searchFinal[sceneIdx*3 + obIdx].push_back(std::make_pair(rotErr, time));
			transErr_searchFinal[sceneIdx*3 + obIdx].push_back(std::make_pair(transErr, time));

			ofstream plotFile;
			plotFile.open((currScene->scenePath + "debug_search/plotData_" + currScene->sceneObjs[obIdx]->objName + ".txt").c_str(), std::ofstream::out|std::ofstream::app);
			plotFile << rotErr << " " << transErr << " " << emdErr << " " << time << " " << score << std::endl;
			plotFile.close();
		}

		if(evalEMD){
			getRangeEMDComputation(currScene->sceneObjs[obIdx]->pclSegment, currScene->camPose, xrange, yrange, zrange);
			utilities::getEMDError(testPose, groundTruth[obIdx].second, currScene->sceneObjs[obIdx]->pclModel, emdErr, xrange, yrange, zrange);
		}
		emdErr_searchFinal[sceneIdx*3 + obIdx].push_back(std::make_pair(emdErr, time));

		super4pcsFile.close();
		timeFile.close();
	}

}

/********************************* function: getPCAError ***********************************************
*******************************************************************************************************/
void Evaluate::getPCAError(scene::Scene *currScene, int sceneIdx){
	std::cout << "Evaluate::getPCAError" << std::endl;

	for(int obIdx=0;obIdx<currScene->sceneObjs.size();obIdx++){
		ifstream pcaFile;
		Eigen::Matrix4f testPose;
		float rotErr, transErr, emdErr=0;
		std::string bestState;
		std::pair<float, float> xrange, yrange, zrange;

		testPose.setIdentity();
		pcaFile.open((currScene->scenePath + "RCNN_PCA_" + currScene->sceneObjs[obIdx]->objName + ".txt").c_str(), std::ifstream::in);
		
		while(pcaFile >> testPose(0,0) >> testPose(0,1) >> testPose(0,2) >> testPose(0,3) 
					 >> testPose(1,0) >> testPose(1,1) >> testPose(1,2) >> testPose(1,3)
					 >> testPose(2,0) >> testPose(2,1) >> testPose(2,2) >> testPose(2,3)){
			

			utilities::getPoseError(testPose, groundTruth[obIdx].second, groundTruth[obIdx].first->symInfo, rotErr, transErr);
			rotErr_PCA[sceneIdx*3 + obIdx] = rotErr;
			transErr_PCA[sceneIdx*3 + obIdx] = transErr;

			ofstream plotFile;
			plotFile.open((currScene->scenePath + "PCAErr_" + currScene->sceneObjs[obIdx]->objName + ".txt").c_str(), std::ofstream::out);
			plotFile << rotErr << " " << transErr << " " << emdErr << std::endl;
			plotFile.close();

			ofstream resFile;
			resFile.open ("/home/chaitanya/PoseDataset17/resultPCA.txt", std::ofstream::out|std::ofstream::app);
			resFile <<"getSuper4pcsError: " << sceneIdx*3 + obIdx << " " << rotErr << " " << transErr << " " << emdErr << std::endl;
			resFile.close();

		}

		if(evalEMD){
			getRangeEMDComputation(currScene->sceneObjs[obIdx]->pclSegment, currScene->camPose, xrange, yrange, zrange);
			utilities::getEMDError(testPose, groundTruth[obIdx].second, currScene->sceneObjs[obIdx]->pclModel, emdErr, xrange, yrange, zrange);
		}
		emdErr_PCA[sceneIdx*3 + obIdx] = emdErr;

		pcaFile.close();
	}

}

/********************************* function: getAllHypoError *******************************************
*******************************************************************************************************/
void Evaluate::getAllHypoError(scene::Scene *currScene, int sceneIdx){
	std::cout << "Evaluate::getAllHypoError" << std::endl;

	for(int obIdx=0;obIdx<currScene->sceneObjs.size();obIdx++){
		ifstream super4pcsFile;
		Eigen::Matrix4f testPose, bestRotPose, bestTransPose;
		float rotErr, transErr, emdErr = 0;
		std::pair<float, float> xrange, yrange, zrange;

		testPose.setIdentity();
		super4pcsFile.open((currScene->scenePath + "debug_super4PCS/allPose_" + currScene->sceneObjs[obIdx]->objName + ".txt").c_str(), std::ifstream::in);
		
		int count = 0;
		while(super4pcsFile >> testPose(0,0) >> testPose(0,1) >> testPose(0,2) >> testPose(0,3) 
					 >> testPose(1,0) >> testPose(1,1) >> testPose(1,2) >> testPose(1,3)
					 >> testPose(2,0) >> testPose(2,1) >> testPose(2,2) >> testPose(2,3)){
			
			utilities::getPoseError(testPose, groundTruth[obIdx].second, groundTruth[obIdx].first->symInfo, rotErr, transErr);
			
			std::cout << "obj id: " << sceneIdx*3 + obIdx << " count: " << count << std::endl;
			count++;

			if(rotErr < rotErr_allhypothesisMinRot[sceneIdx*3 + obIdx]){
				rotErr_allhypothesisMinRot[sceneIdx*3 + obIdx] = rotErr;
				transErr_allhypothesisMinRot[sceneIdx*3 + obIdx] = transErr;
				bestRotPose = testPose;
			}

			if(transErr < transErr_allhypothesisMintrans[sceneIdx*3 + obIdx]){
				rotErr_allhypothesisMintrans[sceneIdx*3 + obIdx] = rotErr;
				transErr_allhypothesisMintrans[sceneIdx*3 + obIdx] = transErr;
				bestTransPose = testPose;
			}
		}

		if(evalEMD){
			getRangeEMDComputation(currScene->sceneObjs[obIdx]->pclSegment, currScene->camPose, xrange, yrange, zrange);
			utilities::getEMDError(bestRotPose, groundTruth[obIdx].second, currScene->sceneObjs[obIdx]->pclModel, emdErr, xrange, yrange, zrange);
			emdErr_allhypothesisMinRot[sceneIdx*3 + obIdx] = emdErr;
			utilities::getEMDError(bestTransPose, groundTruth[obIdx].second, currScene->sceneObjs[obIdx]->pclModel, emdErr, xrange, yrange, zrange);
			emdErr_allhypothesisMintrans[sceneIdx*3 + obIdx] = emdErr;
		}

		ofstream resFile;
		resFile.open ("/home/chaitanya/PoseDataset17/resultAllHypo.txt", std::ofstream::out|std::ofstream::app);
		resFile <<"getSuper4pcsError: " << sceneIdx*3 + obIdx << " " << rotErr_allhypothesisMinRot[sceneIdx*3 + obIdx]
			    << " " << transErr_allhypothesisMinRot[sceneIdx*3 + obIdx] << std::endl;
		resFile.close();

		super4pcsFile.close();
	}
}

/********************************* function: getClusterHypoError ***************************************
*******************************************************************************************************/
void Evaluate::getClusterHypoError(scene::Scene *currScene, int sceneIdx){
	std::cout << "Evaluate::getClusterHypoError" << std::endl;

	for(int obIdx=0;obIdx<currScene->sceneObjs.size();obIdx++){
		ifstream super4pcsFile;
		Eigen::Matrix4f testPose, bestRotPose, bestTransPose;
		float rotErr, transErr, emdErr=0;
		std::pair<float, float> xrange, yrange, zrange;

		testPose.setIdentity();
		super4pcsFile.open((currScene->scenePath + "debug_super4PCS/clusterPose_" + currScene->sceneObjs[obIdx]->objName + ".txt").c_str(), std::ifstream::in);

		while(super4pcsFile >> testPose(0,0) >> testPose(0,1) >> testPose(0,2) >> testPose(0,3) 
					 >> testPose(1,0) >> testPose(1,1) >> testPose(1,2) >> testPose(1,3)
					 >> testPose(2,0) >> testPose(2,1) >> testPose(2,2) >> testPose(2,3)){
			
			utilities::getPoseError(testPose, groundTruth[obIdx].second, groundTruth[obIdx].first->symInfo, rotErr, transErr);
			
			if(rotErr < rotErr_clusterhypothesisMinRot[sceneIdx*3 + obIdx]){
				rotErr_clusterhypothesisMinRot[sceneIdx*3 + obIdx] = rotErr;
				transErr_clusterhypothesisMinRot[sceneIdx*3 + obIdx] = transErr;
				bestRotPose = testPose;
			}

			if(transErr < transErr_clusterhypothesisMintrans[sceneIdx*3 + obIdx]){
				rotErr_clusterhypothesisMintrans[sceneIdx*3 + obIdx] = rotErr;
				transErr_clusterhypothesisMintrans[sceneIdx*3 + obIdx] = transErr;
				bestTransPose = testPose;
			}
		}

		if(evalEMD){
			getRangeEMDComputation(currScene->sceneObjs[obIdx]->pclSegment, currScene->camPose, xrange, yrange, zrange);
			utilities::getEMDError(bestRotPose, groundTruth[obIdx].second, currScene->sceneObjs[obIdx]->pclModel, emdErr, xrange, yrange, zrange);
			emdErr_clusterhypothesisMinRot[sceneIdx*3 + obIdx] = emdErr;
			utilities::getEMDError(bestTransPose, groundTruth[obIdx].second, currScene->sceneObjs[obIdx]->pclModel, emdErr, xrange, yrange, zrange);
			emdErr_clusterhypothesisMintrans[sceneIdx*3 + obIdx] = emdErr;
		}

		ofstream resFile;
		resFile.open ("/home/chaitanya/PoseDataset17/resultClusterHypo.txt", std::ofstream::out|std::ofstream::app);
		resFile <<"getSuper4pcsError: " << sceneIdx*3 + obIdx << " " << rotErr_clusterhypothesisMinRot[sceneIdx*3 + obIdx]
			    << " " << transErr_clusterhypothesisMinRot[sceneIdx*3 + obIdx] << std::endl;
		resFile.close();

		super4pcsFile.close();
	}
}

/********************************* function: readGroundTruth *******************************************
*******************************************************************************************************/

void Evaluate::readGroundTruth(scene::Scene *currScene){
	std::cout << "Evaluate::readGroundTruth" << std::endl;

	groundTruth.clear();
	for(int i=0;i<currScene->sceneObjs.size();i++){
		ifstream gtPoseFile;
		Eigen::Matrix4f gtPose;
		gtPose.setIdentity();
		gtPoseFile.open((currScene->scenePath + "gt_pose_" + currScene->sceneObjs[i]->objName + ".txt").c_str(), std::ifstream::in);
		gtPoseFile >> gtPose(0,0) >> gtPose(0,1) >> gtPose(0,2) >> gtPose(0,3) 
				 >> gtPose(1,0) >> gtPose(1,1) >> gtPose(1,2) >> gtPose(1,3)
				 >> gtPose(2,0) >> gtPose(2,1) >> gtPose(2,2) >> gtPose(2,3);
		gtPoseFile.close();
		groundTruth.push_back(std::make_pair(currScene->sceneObjs[i], gtPose));
	}
}

/********************************* function: writeResults **********************************************
*******************************************************************************************************/

void Evaluate::writeResults(){
	std::cout << "Evaluate::writeResults" << std::endl;

	ofstream resFile;

	float meanrotErr_super4pcs = 0;
	float meantransErr_super4pcs = 0;
	float meanemdErr_super4pcs = 0;

	float meanrotErr_super4pcsICP = 0;
	float meantransErr_super4pcsICP = 0;
	float meanemdErr_super4pcsICP = 0;

	float meanrotErr_allhypothesisMinRot = 0;
	float meantransErr_allhypothesisMinRot = 0;
	float meanemdErr_allhypothesisMinRot = 0;

	float meanrotErr_allhypothesisMintrans = 0;
	float meantransErr_allhypothesisMintrans = 0;
	float meanemdErr_allhypothesisMintrans = 0;

	float meanrotErr_clusterhypothesisMinRot = 0;
	float meantransErr_clusterhypothesisMinRot = 0;
	float meanemdErr_clusterhypothesisMinRot = 0;

	float meanrotErr_clusterhypothesisMintrans = 0;
	float meantransErr_clusterhypothesisMintrans = 0;
	float meanemdErr_clusterhypothesisMintrans = 0;

	float meanrotErr_searchFinal = 0;
	float meantransErr_searchFinal = 0;
	float meanemdErr_searchFinal = 0;

	float meanrotErr_PCA = 0;
	float meantransErr_PCA = 0;
	float meanemdErr_PCA = 0;

	resFile.open ("/home/chaitanya/PoseDataset17/result.txt", std::ofstream::out);

	for(int i=0;i<DATASET_SIZE;i++){
		meanrotErr_super4pcs += rotErr_super4pcs[i];
		meantransErr_super4pcs += transErr_super4pcs[i];
		meanemdErr_super4pcs += emdErr_super4pcs[i];

		meanrotErr_PCA += rotErr_PCA[i];
		meantransErr_PCA += transErr_PCA[i];
		meanemdErr_PCA += emdErr_PCA[i];

		meanrotErr_super4pcsICP += rotErr_super4pcsICP[i];
		meantransErr_super4pcsICP += transErr_super4pcsICP[i];
		meanemdErr_super4pcsICP += emdErr_super4pcsICP[i];

		meanrotErr_allhypothesisMinRot += rotErr_allhypothesisMinRot[i];
		meantransErr_allhypothesisMinRot += transErr_allhypothesisMinRot[i];
		meanemdErr_allhypothesisMinRot += emdErr_allhypothesisMinRot[i];

		meanrotErr_allhypothesisMintrans += rotErr_allhypothesisMintrans[i];
		meantransErr_allhypothesisMintrans += transErr_allhypothesisMintrans[i];
		meanemdErr_allhypothesisMintrans += emdErr_allhypothesisMintrans[i];

		meanrotErr_clusterhypothesisMinRot += rotErr_clusterhypothesisMinRot[i];
		meantransErr_clusterhypothesisMinRot += transErr_clusterhypothesisMinRot[i];
		meanemdErr_clusterhypothesisMinRot += emdErr_clusterhypothesisMinRot[i];

		meanrotErr_clusterhypothesisMintrans += rotErr_clusterhypothesisMintrans[i];
		meantransErr_clusterhypothesisMintrans += transErr_clusterhypothesisMintrans[i];
		meanemdErr_clusterhypothesisMintrans += emdErr_clusterhypothesisMintrans[i];

		meanrotErr_searchFinal += rotErr_searchFinal[i][rotErr_searchFinal[i].size()-1].first;
		meantransErr_searchFinal += transErr_searchFinal[i][transErr_searchFinal[i].size()-1].first;
		meanemdErr_searchFinal += emdErr_searchFinal[i][transErr_searchFinal[i].size()-1].first;

		ofstream pSearchFile;
		pSearchFile.open("/home/chaitanya/PoseDataset17/resultSearch.txt", std::ofstream::out|std::ofstream::app);
		pSearchFile << rotErr_searchFinal[i][rotErr_searchFinal[i].size()-1].first << " " << 
					transErr_searchFinal[i][transErr_searchFinal[i].size()-1].first << " " <<
					emdErr_searchFinal[i][transErr_searchFinal[i].size()-1].first << std::endl;
		pSearchFile.close();
	}

	meanrotErr_super4pcs /= DATASET_SIZE;
	meantransErr_super4pcs /= DATASET_SIZE;
	meanemdErr_super4pcs /= DATASET_SIZE;

	meanrotErr_super4pcsICP /= DATASET_SIZE;
	meantransErr_super4pcsICP /= DATASET_SIZE;
	meanemdErr_super4pcsICP /= DATASET_SIZE;

	meanrotErr_allhypothesisMinRot /= DATASET_SIZE;
	meantransErr_allhypothesisMinRot /= DATASET_SIZE;
	meanemdErr_allhypothesisMinRot /= DATASET_SIZE;

	meanrotErr_allhypothesisMintrans /= DATASET_SIZE;
	meantransErr_allhypothesisMintrans /= DATASET_SIZE;
	meanemdErr_allhypothesisMintrans /= DATASET_SIZE;

	meanrotErr_clusterhypothesisMinRot /= DATASET_SIZE;
	meantransErr_clusterhypothesisMinRot /= DATASET_SIZE;
	meanemdErr_clusterhypothesisMinRot /= DATASET_SIZE;

	meanrotErr_clusterhypothesisMintrans /= DATASET_SIZE;
	meantransErr_clusterhypothesisMintrans /= DATASET_SIZE;
	meanemdErr_clusterhypothesisMintrans /= DATASET_SIZE;

	meanrotErr_searchFinal /= DATASET_SIZE;
	meantransErr_searchFinal /= DATASET_SIZE;
	meanemdErr_searchFinal /= DATASET_SIZE;

	meanrotErr_PCA /= DATASET_SIZE;
	meantransErr_PCA /= DATASET_SIZE;
	meanemdErr_PCA /= DATASET_SIZE;

	resFile << "meanrotErr_super4pcs: " << meanrotErr_super4pcs << std::endl;
	resFile << "meantransErr_super4pcs: " << meantransErr_super4pcs << std::endl;
	resFile << "meanemdErr_super4pcs: " << meanemdErr_super4pcs << std::endl;
	resFile << std::endl;

	resFile << "meanrotErr_super4pcsICP: " << meanrotErr_super4pcsICP << std::endl;
	resFile << "meantransErr_super4pcsICP: " << meantransErr_super4pcsICP << std::endl;
	resFile << "meanemdErr_super4pcsICP: " << meanemdErr_super4pcsICP << std::endl;
	resFile << std::endl;

	resFile << "meanrotErr_allhypothesisMinRot: " << meanrotErr_allhypothesisMinRot << std::endl;
	resFile << "meantransErr_allhypothesisMinRot: " << meantransErr_allhypothesisMinRot << std::endl;
	resFile << "meanemdErr_allhypothesisMinRot: " << meanemdErr_allhypothesisMinRot << std::endl;
	resFile << std::endl;

	resFile << "meanrotErr_allhypothesisMintrans: " << meanrotErr_allhypothesisMintrans << std::endl;
	resFile << "meantransErr_allhypothesisMintrans: " << meantransErr_allhypothesisMintrans << std::endl;
	resFile << "meanemdErr_allhypothesisMintrans: " << meanemdErr_allhypothesisMintrans << std::endl;
	resFile << std::endl;

	resFile << "meanrotErr_clusterhypothesisMinRot: " << meanrotErr_clusterhypothesisMinRot << std::endl;
	resFile << "meantransErr_clusterhypothesisMinRot: " << meantransErr_clusterhypothesisMinRot << std::endl;
	resFile << "meanemdErr_clusterhypothesisMinRot: " << meanemdErr_clusterhypothesisMinRot << std::endl;
	resFile << std::endl;

	resFile << "meanrotErr_clusterhypothesisMintrans: " << meanrotErr_clusterhypothesisMintrans << std::endl;
	resFile << "meantransErr_clusterhypothesisMintrans: " << meantransErr_clusterhypothesisMintrans << std::endl;
	resFile << "meanemdErr_clusterhypothesisMintrans: " << meanemdErr_clusterhypothesisMintrans << std::endl;
	resFile << std::endl;

	resFile << "meanrotErr_searchFinal: " << meanrotErr_searchFinal << std::endl;
	resFile << "meantransErr_searchFinal: " << meantransErr_searchFinal << std::endl;
	resFile << "meanemdErr_searchFinal: " << meanemdErr_searchFinal << std::endl;
	resFile << std::endl;

	resFile << "meanrotErr_PCA: " << meanrotErr_PCA << std::endl;
	resFile << "meantransErr_PCA: " << meantransErr_PCA << std::endl;
	resFile << "meanemdErr_PCA: " << meanemdErr_PCA << std::endl;
	resFile << std::endl;

	resFile.close();
}

}