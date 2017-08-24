#include <Scene.hpp>
#include <State.hpp>
#include <fstream>

#include <detection_package/UpdateActiveListFrame.h>
#include <detection_package/UpdateBbox.h>

// Super4PCS package
int getProbableTransformsSuper4PCS(std::string input1, std::string input2, Eigen::Isometry3d &bestPose, 
              float &bestscore, std::vector< std::pair <Eigen::Isometry3d, float> > &allPose);

namespace scene{

	/********************************* function: constructor ***********************************************
	*******************************************************************************************************/

	Scene::Scene(std::string scenePath){
		srand (time(NULL));

		// Reading objects file and add the scene objects
		std::ifstream filein((scenePath + "objects.txt").c_str());
		std::string line;
		while(std::getline(filein, line)){
			for(int i=0;i<Objects.size();i++){
				if(!line.compare(Objects[i]->objName))
					sceneObjs.push_back(Objects[i]);
			}
		}
		this->scenePath = scenePath;
		numObjects = sceneObjs.size();
		filein.close();

		// Reading camera pose
		camPose = Eigen::Matrix4f::Zero(4,4);
		filein.open ((scenePath + "cameraExtrinsic.txt").c_str(), std::ifstream::in);
		for(int i=0; i<4; i++)
			for(int j=0; j<4; j++)
	  			filein >> camPose(i,j);
	  	filein.close();

	  	// Reading camera intrinsic matrix
		camIntrinsic = Eigen::Matrix3f::Zero(3,3);
		filein.open ((scenePath + "cameraIntinsic.txt").c_str(), std::ifstream::in);
		for(int i=0; i<3; i++)
			for(int j=0; j<3; j++)
	  			filein >> camIntrinsic(i,j);
	  	filein.close();

	  	colorImage = cv::imread(scenePath + "frame-000000.color.png", CV_LOAD_IMAGE_COLOR);
	    utilities::readDepthImage(depthImage, scenePath + "frame-000000.depth.png");
	} 

	/********************************* function: performRCNNDetection **************************************
	*******************************************************************************************************/

	void Scene::performRCNNDetection(){
		ros::NodeHandle n;
		ros::ServiceClient clientlist = n.serviceClient<detection_package::UpdateActiveListFrame>("/update_active_list_and_frame");
		ros::ServiceClient clientbox = n.serviceClient<detection_package::UpdateBbox>("/update_bbox");
		detection_package::UpdateActiveListFrame listsrv;
		detection_package::UpdateBbox boxsrv;

		// Update object list
		for(int i=0;i<numObjects;i++){
			listsrv.request.active_list.push_back(sceneObjs[i]->objIdx);
			listsrv.request.active_frame = "000000";
		    if(clientlist.call(listsrv))
    			ROS_INFO("Scene::performRCNNDetection, object: %d", i+1);
  		  	else{
    			ROS_ERROR("Failed to call service UpdateActiveListFrame");
    			exit(1);
    	  	}
    	}

    	// Calling R-CNN
    	boxsrv.request.scene_path = scenePath;
    	if (clientbox.call(boxsrv))
      		for(int i=0;i<numObjects;i++)
        		sceneObjs[i]->bbox	= cv::Rect(boxsrv.response.tl_x[i], boxsrv.response.tl_y[i], boxsrv.response.br_x[i] - boxsrv.response.tl_x[i],
                    							boxsrv.response.br_y[i] - boxsrv.response.tl_y[i]);
        else{
      		ROS_ERROR("Failed to call service UpdateBbox");
      		exit(1);
    	}
	}

	/********************************* function: get3DSegments *********************************************
	*******************************************************************************************************/

	void Scene::get3DSegments(){
		for(int i=0;i<numObjects;i++){
			std::cout << "Scene::get3DSegments: " << sceneObjs[i]->objName << std::endl;
			cv::Mat mask = cv::Mat::zeros(colorImage.rows, colorImage.cols, CV_32FC1);
			mask(sceneObjs[i]->bbox) = 1.0;
			cv::Mat objDepth = depthImage.mul(mask);
			sceneObjs[i]->pclSegment = PointCloud::Ptr(new PointCloud);
			sceneObjs[i]->pclSegmentDense = PointCloud::Ptr(new PointCloud);
			utilities::convert3dUnOrganized(objDepth, camIntrinsic, sceneObjs[i]->pclSegment);

			pcl::VoxelGrid<pcl::PointXYZ> sor;
			sor.setInputCloud (sceneObjs[i]->pclSegment);
			sor.setLeafSize (0.005f, 0.005f, 0.005f);
			sor.filter (*sceneObjs[i]->pclSegment);
		}
	}

	/********************************* function: removeTable ***********************************************
	References: http://pointclouds.org/documentation/tutorials/planar_segmentation.php,
	http://pointclouds.org/documentation/tutorials/extract_indices.php
	*******************************************************************************************************/

	void Scene::removeTable(){
		PointCloud::Ptr SampledSceneCloud(new PointCloud);
		sceneCloud = PointCloud::Ptr(new PointCloud);
		utilities::convert3dOrganized(depthImage, camIntrinsic, sceneCloud);

		std::string input1 = scenePath + "debug/scene.ply";
		std::cout << input1 <<std::endl;
		pcl::io::savePLYFile(input1, *sceneCloud);
		
		// Creating the filtering object: downsample the dataset using a leaf size of 0.5cm
		pcl::VoxelGrid<pcl::PointXYZ> sor;
		sor.setInputCloud (sceneCloud);
		sor.setLeafSize (0.005f, 0.005f, 0.005f);
		sor.filter (*SampledSceneCloud);

		// Plane Fitting
		pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
		pcl::SACSegmentation<pcl::PointXYZ> seg;
		seg.setOptimizeCoefficients (true);
		seg.setModelType (pcl::SACMODEL_PLANE);
		seg.setMethodType (pcl::SAC_MSAC);
		seg.setDistanceThreshold (0.005);
		seg.setMaxIterations (1000);
		seg.setInputCloud (SampledSceneCloud);
		seg.segment (*inliers, *coefficients);

		std::cout << "table co-ordinates are: " << coefficients->values[0] << coefficients->values[1] <<
														coefficients->values[2] << coefficients->values[3] <<std::endl;
		int imgWidth = depthImage.cols;
		int imgHeight = depthImage.rows;

		for(int u=0; u<imgHeight; u++)
			for(int v=0; v<imgWidth; v++){
				float depth = depthImage.at<float>(u,v);
				pcl::PointXYZ pt;
				pt.x = (float)((v - camIntrinsic(0,2)) * depth / camIntrinsic(0,0));
				pt.y = (float)((u - camIntrinsic(1,2)) * depth / camIntrinsic(1,1));
				pt.z = depth;
				double dist = pcl::pointToPlaneDistance(pt, coefficients->values[0], coefficients->values[1],
														coefficients->values[2], coefficients->values[3]);
				if(dist<0.005)
					depthImage.at<float>(u,v) = 0;
		}
		utilities::writeDepthImage(depthImage, scenePath + "debug/scene.png");
	}

	/********************************* function: getOrder **************************************************
	*******************************************************************************************************/

	void Scene::getOrder(){
		objOrder = sceneObjs;
	}

	/********************************* function: readGroundTruth *******************************************
	*******************************************************************************************************/
	void Scene::readGroundTruth(){
		for(int i=0;i<objOrder.size();i++){
			ifstream gtPoseFile;
			Eigen::Matrix4f gtPose;
			gtPose.setIdentity();
			gtPoseFile.open((scenePath + "gt_pose_" + objOrder[i]->objName + ".txt").c_str(), std::ifstream::in);
			gtPoseFile >> gtPose(0,0) >> gtPose(0,1) >> gtPose(0,2) >> gtPose(0,3) 
					 >> gtPose(1,0) >> gtPose(1,1) >> gtPose(1,2) >> gtPose(1,3)
					 >> gtPose(2,0) >> gtPose(2,1) >> gtPose(2,2) >> gtPose(2,3);
			gtPoseFile.close();
			groundTruth.push_back(std::make_pair(objOrder[i], gtPose));
		}
	}
	
	/********************************* function: getHypothesis *********************************************
	*******************************************************************************************************/
#ifdef DBG_SUPER4PCS
	void Scene::getHypothesis(apc_objects::APCObjects* obj, PointCloud::Ptr pclSegment, PointCloud::Ptr pclModel, 
		std::vector< std::pair <Eigen::Isometry3d, float> > &allPose, Eigen::Matrix4f gtPose){
#else
	void Scene::getHypothesis(apc_objects::APCObjects* obj, PointCloud::Ptr pclSegment, PointCloud::Ptr pclModel, 
		std::vector< std::pair <Eigen::Isometry3d, float> > &allPose){
#endif
		const clock_t begin_time = clock();
		std::string input1 = scenePath + "debug/pclSegment_" + obj->objName + ".ply";
		std::string input2 = scenePath + "debug/pclModel_" + obj->objName + ".ply";
		pcl::io::savePLYFile(input1, *pclSegment);
		pcl::io::savePLYFile(input2, *pclModel);

		Eigen::Isometry3d bestPose;
		float bestscore = 0;
		getProbableTransformsSuper4PCS(input1, input2, bestPose, bestscore, allPose);
		max4PCSPose.push_back(std::make_pair(bestPose, bestscore));
		
		std::cout << "object hypothesis count: " << allPose.size() << std::endl;
		std::cout << "bestScore: " << bestscore <<std::endl;
		std::cout << "Scene::getHypothesis: Super4PCS time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
		std::cout << "###################################################" <<std::endl;

	#ifdef DBG_SUPER4PCS
		// get the closest to ground truth from set of all hypothesis
		float rotErr, transErr;
		float minRotErr_rot = INT_MAX;
		float minRotErr_trans = INT_MAX;

		float minTransErr_rot = INT_MAX;
		float minTransErr_trans = INT_MAX;

		cv::Mat points(allPose.size(), 6, CV_32F);
		int bestGTPoseIdx = -1;
		for(int ii = 0; ii < allPose.size(); ii++){
			Eigen::Matrix4f pose;
			utilities::convertToMatrix(allPose[ii].first, pose);
			utilities::convertToWorld(pose, camPose);
			utilities::addToCVMat(pose, points, ii);
			utilities::getPoseError(pose, gtPose, obj->symInfo, rotErr, transErr);
			if(rotErr < minRotErr_rot){
				minRotErr_rot = rotErr;
				minRotErr_trans = transErr;
				bestGTPoseIdx = ii;
			}
			if(transErr < minTransErr_trans){
				minTransErr_rot = rotErr;
				minTransErr_trans = transErr;
			}
		}

		// trial test:: get 9 random and 1 best hypothesis to search over
		std::vector< std::pair <Eigen::Isometry3d, float> > subsetPose;
		subsetPose.push_back(allPose[bestGTPoseIdx]);
		// for (int i=0; i<9; ++i) {
		// 	int number = rand() % allPose.size();
		// 	subsetPose.push_back(allPose[number]);
		// }
		unconditionedHypothesis.push_back(subsetPose);

		ofstream statsFile;
		statsFile.open ((scenePath + "debug/stats_" + obj->objName + ".txt").c_str(), std::ofstream::out | std::ofstream::app);
		statsFile << "allHypBestRotation_RotErr: " << minRotErr_rot << std::endl;
		statsFile << "allHypBestRotation_TransErr: " << minRotErr_trans << std::endl;
		statsFile << "allHypBestTranslation_RotErr: " << minTransErr_rot << std::endl;
		statsFile << "allHypBestTranslation_TransErr: " << minTransErr_trans << std::endl;
		statsFile.close();

		// k-means clustering of poses
		int k = 50;
		cv::Mat colMean, colMeanRepAll, colMeanRepK;
		cv::reduce(points, colMean, 0, CV_REDUCE_AVG);
		cv::repeat(colMean, allPose.size(), 1, colMeanRepAll);
		cv::repeat(colMean, k, 1, colMeanRepK);
		points = points/colMeanRepAll;

		cv::Mat clusterIndices;
		cv::Mat clusterCenters;
		cv::kmeans(points, /* num clusters */ k, clusterIndices, cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 1000, 0.001)
			, /*int attempts*/ 10, cv::KMEANS_PP_CENTERS, clusterCenters);

		float minRotErrCluster_rot = INT_MAX;
		float minRotErrCluster_trans = INT_MAX;

		float minTransErrCluster_rot = INT_MAX;
		float minTransErrCluster_trans = INT_MAX;

		int bestClusterIdx = -1;
		clusterCenters = clusterCenters.mul(colMeanRepK);
		for(int ii = 0; ii < k; ii++){
			Eigen::Matrix4f tmpPose;
			utilities::convert6DToMatrix(tmpPose, clusterCenters, ii);
			utilities::getPoseError(tmpPose, gtPose, obj->symInfo, rotErr, transErr);
			if(rotErr < minRotErrCluster_rot){
				minRotErrCluster_rot = rotErr;
				minRotErrCluster_trans = transErr;
				bestClusterIdx = ii;
			}
			if(transErr < minTransErrCluster_trans){
				minTransErrCluster_rot = rotErr;
				minTransErrCluster_trans = transErr;
			}
		}

		statsFile.open ((scenePath + "debug/stats_" + obj->objName + ".txt").c_str(), std::ofstream::out | std::ofstream::app);
		statsFile << "ClusterHypBestRotation_RotErr: " << minRotErrCluster_rot << std::endl;
		statsFile << "ClusterHypBestRotation_TransErr: " << minRotErrCluster_trans << std::endl;
		statsFile << "ClusterHypBestTranslation_RotErr: " << minTransErrCluster_rot << std::endl;
		statsFile << "ClusterHypBestTranslation_TransErr: " << minTransErrCluster_trans << std::endl;
		statsFile.close();

		// within cluster search
		// float minRotErrWithinCluster_rot = INT_MAX;
		// float minRotErrWithinCluster_trans = INT_MAX;

		// float minTransErrWithinCluster_rot = INT_MAX;
		// float minTransErrWithinCluster_trans = INT_MAX;

		// points = points.mul(colMeanRepAll);
		// for(int ii = 0; ii < points.rows; ii++){
		// 	if(clusterIndices.at<int>(ii) == bestClusterIdx){
		// 		Eigen::Matrix4f tmpPose;
		// 		utilities::convert6DToMatrix(tmpPose, points, ii);
		// 		utilities::getPoseError(tmpPose, gtPose, obj->symInfo, rotErr, transErr);
		// 		if(rotErr < minRotErrWithinCluster_rot){
		// 			minRotErrWithinCluster_rot = rotErr;
		// 			minRotErrWithinCluster_trans = transErr;
		// 		}
		// 		if(transErr < minTransErrWithinCluster_trans){
		// 			minTransErrWithinCluster_rot = rotErr;
		// 			minTransErrWithinCluster_trans = transErr;
		// 		}
		// 	}
		// }

		// statsFile.open ((scenePath + "debug/stats_" + obj->objName + ".txt").c_str(), std::ofstream::out | std::ofstream::app);
		// statsFile << "minRotErrWithinCluster_rot: " << minRotErrWithinCluster_rot << std::endl;
		// statsFile << "minRotErrWithinCluster_trans: " << minRotErrWithinCluster_trans << std::endl;
		// statsFile << "minTransErrWithinCluster_rot: " << minTransErrWithinCluster_rot << std::endl;
		// statsFile << "minTransErrWithinCluster_trans: " << minTransErrWithinCluster_trans << std::endl;
		// statsFile.close();
		
	#endif
	}

	/********************************* function: getUnconditionedHypothesis ********************************
	*******************************************************************************************************/

	void Scene::getUnconditionedHypothesis(){
	#ifdef DBG_SUPER4PCS
		readGroundTruth();
		for(int i=0;i<objOrder.size();i++){
			std::vector< std::pair <Eigen::Isometry3d, float> > allPose;
			getHypothesis(objOrder[i], objOrder[i]->pclSegment, objOrder[i]->pclModel, allPose, groundTruth[i].second);
		}

		// render closest to ground truth pose
		state::State* bestState = new state::State(numObjects);
		bestState->updateStateId(-2);
		for(int i=0;i<objOrder.size();i++)
			bestState->updateNewObject(objOrder[i], std::make_pair(unconditionedHypothesis[i][0].first, 0.f), numObjects);
		
		cv::Mat depth_image_minGT;
		bestState->render(camPose, scenePath, depth_image_minGT);
		bestState->computeCost(depth_image_minGT, depthImage);
		
		ofstream scoreFile;
		scoreFile.open ((scenePath + "debug/scores.txt").c_str(), std::ofstream::out | std::ofstream::app);
		scoreFile << bestState->stateId << " " << bestState->score << std::endl;
		scoreFile.close();

	#else
		for(int i=0;i<objOrder.size();i++){
			std::vector< std::pair <Eigen::Isometry3d, float> > allPose;
			getHypothesis(objOrder[i], objOrder[i]->pclSegment, objOrder[i]->pclModel, allPose);
			unconditionedHypothesis.push_back(allPose);
		}
	#endif

		// evaluate the best LCP hypothesis returned by Super4PCS
		state::State* max4PCSState = new state::State(0);
		max4PCSState->updateStateId(-1);
		
		physim::PhySim *tmpSim = new physim::PhySim();
		tmpSim->addTable(0.55);
		for(int ii=0; ii<objOrder.size(); ii++)
			tmpSim->initRigidBody(objOrder[ii]->objName);

		for(int i=0;i<objOrder.size();i++){
			max4PCSState->numObjects = i+1;
			max4PCSState->updateNewObject(objOrder[i], std::make_pair(max4PCSPose[i].first, 0.f), max4PCSState->numObjects);
			Eigen::Matrix4f bestPoseMat;
			float rotErr, transErr;
			utilities::convertToMatrix(max4PCSPose[i].first, bestPoseMat);
			utilities::convertToWorld(bestPoseMat, camPose);
			utilities::getPoseError(bestPoseMat, groundTruth[i].second, objOrder[i]->symInfo, rotErr, transErr);

			ofstream statsFile;
			statsFile.open ((scenePath + "debug/stats_" + objOrder[i]->objName + ".txt").c_str(), std::ofstream::out | std::ofstream::app);
			statsFile << "BestLCP_RotErr: " << rotErr << std::endl;
			statsFile << "BestLCP_TransErr: " << transErr << std::endl;
			statsFile.close();

			// perform ICP and evaluate error
			max4PCSState->performTrICP(scenePath, 0.9);
			utilities::convertToMatrix(max4PCSState->objects[max4PCSState->numObjects-1].second, bestPoseMat);
			utilities::convertToWorld(bestPoseMat, camPose);
			utilities::getPoseError(bestPoseMat, groundTruth[i].second, objOrder[i]->symInfo, rotErr, transErr);
			statsFile.open ((scenePath + "debug/stats_" + objOrder[i]->objName + ".txt").c_str(), std::ofstream::out | std::ofstream::app);
			statsFile << "BestLCP_AfterICP_RotErr: " << rotErr << std::endl;
			statsFile << "BestLCP_AfterICP_TransErr: " << transErr << std::endl;
			statsFile.close();

			// perform simulation and evaluate error
			max4PCSState->correctPhysics(tmpSim, camPose, scenePath);
			utilities::convertToMatrix(max4PCSState->objects[max4PCSState->numObjects-1].second, bestPoseMat);
			utilities::convertToWorld(bestPoseMat, camPose);
			utilities::getPoseError(bestPoseMat, groundTruth[i].second, objOrder[i]->symInfo, rotErr, transErr);
			statsFile.open ((scenePath + "debug/stats_" + objOrder[i]->objName + ".txt").c_str(), std::ofstream::out | std::ofstream::app);
			statsFile << "BestLCP_AfterICPSimulation_RotErr: " << rotErr << std::endl;
			statsFile << "BestLCP_AfterICPSimulation_TransErr: " << transErr << std::endl;
			statsFile.close();

		}
		cv::Mat depth_image_minLCP;
		max4PCSState->render(camPose, scenePath, depth_image_minLCP);
		max4PCSState->computeCost(depth_image_minLCP, depthImage);
	}

	/********************************* end of functions ****************************************************
	*******************************************************************************************************/
} // namespace scene