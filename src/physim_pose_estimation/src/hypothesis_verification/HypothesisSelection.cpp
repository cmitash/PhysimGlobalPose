#include <HypothesisSelection.hpp>

namespace hypothesis_selection{

	HypothesisSelection::HypothesisSelection(){

	}

	HypothesisSelection::~HypothesisSelection(){

	}

	bool sortPoses(const std::pair <Eigen::Isometry3d, float> &a,
	              const std::pair <Eigen::Isometry3d, float> &b) {
	    return (a.second > b.second);
	}

	// For StoCS
	// void HypothesisSelection::greedyClustering(scene_cfg::SceneCfg *pCfg, int objId){
	// 	std::vector< std::pair <Eigen::Isometry3d, float> > prunedHypotheses;
		
	// 	// 1: Prune
	// 	float acceptable_fraction = 0.5;
	// 	float best_score = pCfg->pSceneObjects[objId]->hypotheses->bestHypothesis.second;

	// 	std::cout << "best score: " << best_score << std::endl;

	// 	for(auto pose_it : pCfg->pSceneObjects[objId]->hypotheses->hypothesisSet) {
	// 		if(pose_it.second > acceptable_fraction*best_score)
	// 			prunedHypotheses.push_back(pose_it);
	// 	}

	// 	// 2: Sort
	// 	std::cout << "Number of poses to sort: " << 
	// 			prunedHypotheses.size() << std::endl;

	// 	std::sort(prunedHypotheses.begin(), prunedHypotheses.end(), sortPoses);

	// 	std::cout << "Clustering..." << std::endl;

	// 	// 3: Cluster
	// 	for(auto candidate_it: prunedHypotheses) {
	// 		bool inValid = false;
	// 		// std::cout << "LCP score: " << candidate_it.second << std::endl;
	// 		for(auto cluster_it: pCfg->pSceneObjects[objId]->hypotheses->clusteredHypothesisSet) {
	// 			float meanrotErr, transErr;
	// 			Eigen::Matrix4f candidatePose, clusterPose;
	// 			utilities::convertToMatrix(candidate_it.first, candidatePose);
	// 			utilities::convertToMatrix(cluster_it.first, clusterPose);
	// 			utilities::getPoseError(candidatePose, clusterPose, pCfg->pSceneObjects[objId]->pObject->symInfo, 
	// 							meanrotErr, transErr);
	// 			if(meanrotErr < 10 && transErr < 0.02) {
	// 				inValid = true;
	// 				break;
	// 			}
	// 		}

	// 		if(inValid == false)
	// 			pCfg->pSceneObjects[objId]->hypotheses->clusteredHypothesisSet.push_back(candidate_it);
	// 	}

	// 	std::cout << "Clustered set size: " << pCfg->pSceneObjects[objId]->hypotheses->clusteredHypothesisSet.size() << std::endl;
	// }

	// For Hough Transform
	void HypothesisSelection::greedyClustering(scene_cfg::SceneCfg *pCfg, int objId){
		std::vector< std::pair <Eigen::Isometry3d, float> > prunedHypotheses;
		
		// 1: Prune
		float acceptable_fraction = 0.5;
		float best_score = pCfg->pSceneObjects[objId]->hypotheses->bestHypothesis.second;

		std::cout << "best score: " << best_score << std::endl;

		for(auto pose_it : pCfg->pSceneObjects[objId]->hypotheses->hypothesisSet) {
			if(pose_it.second > acceptable_fraction*best_score)
				prunedHypotheses.push_back(pose_it);
		}

		// 2: Sort
		std::cout << "Number of poses to sort: " << 
				prunedHypotheses.size() << std::endl;

		std::sort(prunedHypotheses.begin(), prunedHypotheses.end(), sortPoses);

		std::cout << "Clustering..." << std::endl;

		// 3: Cluster
		for(auto candidate_it: prunedHypotheses) {
			bool inValid = false;
			// std::cout << "LCP score: " << candidate_it.second << std::endl;
			for(auto cluster_it: pCfg->pSceneObjects[objId]->hypotheses->clusteredHypothesisSet) {
				float meanrotErr, transErr;
				Eigen::Matrix4f candidatePose, clusterPose;
				utilities::convertToMatrix(candidate_it.first, candidatePose);
				utilities::convertToMatrix(cluster_it.first, clusterPose);
				utilities::getPoseError(candidatePose, clusterPose, pCfg->pSceneObjects[objId]->pObject->symInfo, 
								meanrotErr, transErr);
				if(meanrotErr < 10 && transErr < 0.02) {
					inValid = true;
					cluster_it.second += candidate_it.second;
					break;
				}
			}

			if(inValid == false){
				pCfg->pSceneObjects[objId]->hypotheses->clusteredHypothesisSet.push_back(candidate_it);
			}
		}

		std::sort(pCfg->pSceneObjects[objId]->hypotheses->clusteredHypothesisSet.begin(), 
					pCfg->pSceneObjects[objId]->hypotheses->clusteredHypothesisSet.end(), sortPoses);

		std::cout << "Clustered set size: " << pCfg->pSceneObjects[objId]->hypotheses->clusteredHypothesisSet.size() << std::endl;
	}

	void LCPSelection::selectBestPoses(scene_cfg::SceneCfg *pCfg){
		for(int ii=0; ii<pCfg->numObjects; ii++){

			// START: perform ICP on best lcp pose
			// pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclSegmentRegistered(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

			// for(int jj=0; jj<pCfg->pSceneObjects[ii]->hypotheses->registered_points.size(); jj++)
			// 	pclSegmentRegistered->points.push_back(pCfg->pSceneObjects[ii]->pclSegment->points[pCfg->pSceneObjects[ii]->hypotheses->registered_points[jj]]);

			// std::string input3 = pCfg->scenePath + "debug_super4PCS/pclSegment_" + pCfg->pSceneObjects[ii]->pObject->objName + "registered_cloud.ply";
			// pcl::io::savePLYFile(input3, *pclSegmentRegistered);

			// PointCloudRGBNormal::Ptr cloud_out(new PointCloudRGBNormal);
			// pcl::transformPointCloud(*pCfg->pSceneObjects[ii]->pObject->pclModel, *cloud_out, pCfg->pSceneObjects[ii]->hypotheses->bestHypothesis.first.matrix());
			// char hypFile[200];
			// sprintf(hypFile, "debug_super4PCS/hypothesis_%d_best.ply", ii);
			// pcl::io::savePLYFile((pCfg->scenePath + hypFile).c_str(), *cloud_out);

			// Eigen::Matrix4f offsetTransform;
			// std::string segPath = pCfg->scenePath + "debug_super4PCS/pclSegment_" + pCfg->pSceneObjects[ii]->pObject->objName + "registered_cloud.ply";
			// utilities::pointMatcherICP((pCfg->scenePath + hypFile).c_str(), segPath, offsetTransform);
			// utilities::invertTransformationMatrix(offsetTransform);

			// Eigen::Matrix4f initTransform = (pCfg->pSceneObjects[ii]->hypotheses->bestHypothesis.first.matrix()).cast <float> ();
			// Eigen::Matrix4f finalTransform = offsetTransform*initTransform;
			// char hypFileFinal[400];
			// PointCloudRGBNormal::Ptr cloud_out_final(new PointCloudRGBNormal);
			// pcl::transformPointCloud(*pCfg->pSceneObjects[ii]->pObject->pclModel, *cloud_out_final, finalTransform);
			// sprintf(hypFileFinal, "debug_super4PCS/hypothesis_%d_best_final.ply", ii);
			// pcl::io::savePLYFile((pCfg->scenePath + hypFileFinal).c_str(), *cloud_out_final);

			// Eigen::Isometry3d bestposeIsometry;
			// utilities::convertToIsometry3d(finalTransform, bestposeIsometry);
			// pCfg->pSceneObjects[ii]->objPose = bestposeIsometry;
			// END: perform ICP on best lcp pose

			// START: select best lcp pose
			pCfg->pSceneObjects[ii]->objPose = pCfg->pSceneObjects[ii]->hypotheses->bestHypothesis.first;
			// END: select best lcp pose

			std::cout << "hypothesis size: " << pCfg->pSceneObjects[ii]->hypotheses->hypothesisSet.size() << std::endl;

			// clock_t clustering_start = clock();

			// greedyClustering(pCfg, ii);

			// START: select best hough pose
			// pCfg->pSceneObjects[ii]->objPose = pCfg->pSceneObjects[ii]->hypotheses->clusteredHypothesisSet[0].first;
			// END: select best hough pose

			// START: perform ICP on best lcp pose
			// PointCloudRGBNormal::Ptr cloud_out(new PointCloudRGBNormal);
			// pcl::transformPointCloud(*pCfg->pSceneObjects[ii]->pObject->pclModel, *cloud_out, pCfg->pSceneObjects[ii]->objPose.matrix());
			// char hypFile[200];
			// sprintf(hypFile, "debug_super4PCS/hypothesis_%d_best.ply", ii);
			// pcl::io::savePLYFile((pCfg->scenePath + hypFile).c_str(), *cloud_out);

			// Eigen::Matrix4f offsetTransform;
			// utilities::performICP(pCfg->pSceneObjects[ii]->pclSegment, cloud_out, offsetTransform);
			// utilities::invertTransformationMatrix(offsetTransform);

			// Eigen::Matrix4f initTransform = (pCfg->pSceneObjects[ii]->objPose.matrix()).cast <float> ();
			// Eigen::Matrix4f finalTransform = offsetTransform*initTransform;
			// char hypFileFinal[400];
			// PointCloudRGBNormal::Ptr cloud_out_final(new PointCloudRGBNormal);
			// pcl::transformPointCloud(*pCfg->pSceneObjects[ii]->pObject->pclModel, *cloud_out_final, finalTransform);
			// sprintf(hypFileFinal, "debug_super4PCS/hypothesis_%d_best_final.ply", ii);
			// pcl::io::savePLYFile((pCfg->scenePath + hypFileFinal).c_str(), *cloud_out_final);

			// Eigen::Isometry3d bestposeIsometry;
			// utilities::convertToIsometry3d(finalTransform, bestposeIsometry);
			// pCfg->pSceneObjects[ii]->objPose = bestposeIsometry;
			// END: perform ICP on best lcp pose

			// float clustering_time = float( clock () - clustering_start ) /  CLOCKS_PER_SEC;
			// ofstream pFile;
			// pFile.open ("/media/chaitanya/DATADRIVE0/datasets/YCB_Video_Dataset/time_clustering.txt", std::ofstream::out | std::ofstream::app);
			// pFile << clustering_time << " " << pCfg->pSceneObjects[ii]->hypotheses->clusteredHypothesisSet.size() << std::endl;
			// pFile.close();

			// int jj=0;
			for(auto it: pCfg->pSceneObjects[ii]->hypotheses->hypothesisSet) {
				// PointCloudRGBNormal::Ptr cloud_out(new PointCloudRGBNormal);
				// pcl::transformPointCloud(*pCfg->pSceneObjects[ii]->pObject->pclModel, *cloud_out, it.first.matrix());
				// char hypFile[200];
				// sprintf(hypFile, "debug_super4PCS/hypothesis_%d_%d.ply", ii, jj);
				// pcl::io::savePLYFile((pCfg->scenePath + hypFile).c_str(), *cloud_out);

				// Eigen::Matrix4f offsetTransform;
				// // utilities::pointToPlaneICP(pCfg->pSceneObjects[ii]->pclSegment, cloud_out, offsetTransform);
				// std::string segPath = pCfg->scenePath + "debug_super4PCS/pclSegment_" + pCfg->pSceneObjects[ii]->pObject->objName + ".ply";
				// utilities::pointMatcherICP(segPath ,(pCfg->scenePath + hypFile).c_str(), offsetTransform);
				// // utilities::invertTransformationMatrix(offsetTransform);

				// Eigen::Matrix4f initTransform = (it.first.matrix()).cast <float> ();
				// Eigen::Matrix4f finalTransform = offsetTransform*initTransform;
				// char hypFileFinal[400];
				// PointCloudRGBNormal::Ptr cloud_out_final(new PointCloudRGBNormal);
				// pcl::transformPointCloud(*pCfg->pSceneObjects[ii]->pObject->pclModel, *cloud_out_final, finalTransform);
				// sprintf(hypFileFinal, "debug_super4PCS/hypothesis_%d_%d_final.ply", ii, jj);
				// pcl::io::savePLYFile((pCfg->scenePath + hypFileFinal).c_str(), *cloud_out_final);

			    Eigen::Matrix4f finalPoseMat;
			    Eigen::Isometry3d finalPoseIsometric;

			    // Convert the pose to global frame
			    utilities::convertToMatrix(it.first, finalPoseMat);
			    utilities::convertToWorld(finalPoseMat, pCfg->camPose);
			    utilities::convertToIsometry3d(finalPoseMat, finalPoseIsometric);
			    Eigen::Vector3d trans = finalPoseIsometric.translation();
			    Eigen::Quaterniond rot(finalPoseIsometric.rotation());
			    
			    ofstream pFile;
			    pFile.open ((pCfg->scenePath + "debug_super4PCS/" + pCfg->pSceneObjects[ii]->pObject->objName + "_result.txt").c_str(),
			    		 std::ofstream::out | std::ofstream::app);
			    pFile << trans[0] << " " << trans[1] << " " << trans[2] 
			      << " " << rot.w() << " " << rot.x() << " " << rot.y() << " " << rot.z() << std::endl;
			    pFile.close();

			// 	jj++;
			}
		}
	}

	void MCTSSelection::selectBestPoses(scene_cfg::SceneCfg *pCfg){
		std::vector<std::vector<scene_cfg::SceneObjects*> > independentTrees;
		independentTrees.push_back(pCfg->pSceneObjects);

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