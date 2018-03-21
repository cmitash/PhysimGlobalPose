#include <HypothesisSelection.hpp>

namespace hypothesis_selection{

	HypothesisSelection::HypothesisSelection(){

	}

	HypothesisSelection::~HypothesisSelection(){

	}

	void LCPSelection::selectBestPoses(scene_cfg::SceneCfg *pCfg){

		// pCfg->removeTable();

		for(int ii=0; ii<pCfg->numObjects; ii++){
			pCfg->pSceneObjects[ii]->objPose = pCfg->pSceneObjects[ii]->hypotheses->bestHypothesis.first;
			// utilities::performTrICP(pCfg->pSceneObjects[ii]->pclSegment, pCfg->pSceneObjects[ii]->pObject->pclModel, 
			// 				pCfg->pSceneObjects[ii]->objPose, pCfg->pSceneObjects[ii]->objPose, 0.9);
			// std::cout << "hypothesis size: " << pCfg->pSceneObjects[ii]->hypotheses->hypothesisSet.size() << std::endl;
			// int jj=0;
			// for(auto it: pCfg->pSceneObjects[ii]->hypotheses->hypothesisSet) {
			// 	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZRGBNormal>);;
			// 	pcl::transformPointCloud(*pCfg->pSceneObjects[ii]->pObject->pclModel, *cloud_out, it.first.matrix());
			// 	char hypFile[200];
			// 	sprintf(hypFile, "debug_super4PCS/hypothesis_%d_%d.ply", ii, jj);
			// 	pcl::io::savePLYFile((pCfg->scenePath + hypFile).c_str(), *cloud_out);

			//     Eigen::Matrix4f finalPoseMat;
			//     Eigen::Isometry3d finalPoseIsometric;

			//     // Convert the pose to global frame
			//     utilities::convertToMatrix(it.first, finalPoseMat);
			//     utilities::convertToWorld(finalPoseMat, pCfg->camPose);
			//     utilities::convertToIsometry3d(finalPoseMat, finalPoseIsometric);
			//     Eigen::Vector3d trans = finalPoseIsometric.translation();
			//     Eigen::Quaterniond rot(finalPoseIsometric.rotation());

			//     ofstream pFile;
			//     pFile.open ((pCfg->scenePath + "debug_super4PCS/" + pCfg->pSceneObjects[ii]->pObject->objName + "_result.txt").c_str(), std::ofstream::out | std::ofstream::app);
			//     pFile << trans[0] << " " << trans[1] << " " << trans[2] 
			//       << " " << rot.w() << " " << rot.x() << " " << rot.y() << " " << rot.z() << std::endl;
			//     pFile.close();

			// 	jj++;
			// }
		}
	}

	void MCTSSelection::selectBestPoses(scene_cfg::SceneCfg *pCfg){
		std::vector<std::vector<scene_cfg::SceneObjects*> > independentTrees;
		independentTrees.push_back(pCfg->pSceneObjects);

		pCfg->removeTable();
  		pCfg->getTableParams();

		for(int treeIdx=0; treeIdx<independentTrees.size(); treeIdx++){
		    std::vector< std::vector< std::pair <Eigen::Isometry3d, float> > > hypothesis;

		    // iterate over the objects in the tree and create hypothesis set
		    for(int jj=0;jj<independentTrees[treeIdx].size();jj++){
		      hypothesis.push_back(pCfg->pSceneObjects[jj]->hypotheses->hypothesisSet);
		    }

		    uct_search::UCTSearch *UCTSearch = new uct_search::UCTSearch(independentTrees[treeIdx], pCfg->tableParams, hypothesis,
		                                pCfg->scenePath, pCfg->camPose, pCfg->depthImage, treeIdx);
		    UCTSearch->performSearch();

		    for(int ii=0; ii < independentTrees[treeIdx].size(); ii++)
		    	independentTrees[treeIdx][ii]->objPose = UCTSearch->bestState->objects[ii].second;

		    delete UCTSearch;
		 }
	}	

}