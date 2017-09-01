#include <Scene.hpp>
#include <fstream>

#include <detection_package/UpdateActiveListFrame.h>
#include <detection_package/UpdateBbox.h>

#include <opencv2/flann/flann.hpp>

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

	  	lcpThreshold = 0;
	  	finalState = new state::State(0);
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
		if(scenePath.compare("/home/chaitanya/PoseDataset17/table/scene-0001/") == 0){
			std::vector<apc_objects::APCObjects*> tmpobjOrder;
			tmpobjOrder.push_back(objOrder[0]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[1]);
			tmpobjOrder.push_back(objOrder[2]);
			independentTrees.push_back(tmpobjOrder);
		}
		else if(scenePath.compare("/home/chaitanya/PoseDataset17/table/scene-0002/") == 0){
			std::vector<apc_objects::APCObjects*> tmpobjOrder;
			tmpobjOrder.push_back(objOrder[0]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[1]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[2]);
			independentTrees.push_back(tmpobjOrder);
		}
		else if(scenePath.compare("/home/chaitanya/PoseDataset17/table/scene-0003/") == 0){
			std::vector<apc_objects::APCObjects*> tmpobjOrder;
			tmpobjOrder.push_back(objOrder[0]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[1]);
			tmpobjOrder.push_back(objOrder[2]);
			independentTrees.push_back(tmpobjOrder);
		}
		else if(scenePath.compare("/home/chaitanya/PoseDataset17/table/scene-0004/") == 0){
			std::vector<apc_objects::APCObjects*> tmpobjOrder;
			tmpobjOrder.push_back(objOrder[0]);
			tmpobjOrder.push_back(objOrder[1]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[2]);
			independentTrees.push_back(tmpobjOrder);
		}
		else if(scenePath.compare("/home/chaitanya/PoseDataset17/table/scene-0005/") == 0){
			std::vector<apc_objects::APCObjects*> tmpobjOrder;
			tmpobjOrder.push_back(objOrder[0]);
			tmpobjOrder.push_back(objOrder[1]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[2]);
			independentTrees.push_back(tmpobjOrder);
		}
		else if(scenePath.compare("/home/chaitanya/PoseDataset17/table/scene-0006/") == 0){
			std::vector<apc_objects::APCObjects*> tmpobjOrder;
			tmpobjOrder.push_back(objOrder[0]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[1]);
			tmpobjOrder.push_back(objOrder[2]);
			independentTrees.push_back(tmpobjOrder);
		}
		else if(scenePath.compare("/home/chaitanya/PoseDataset17/table/scene-0007/") == 0){
			std::vector<apc_objects::APCObjects*> tmpobjOrder;
			tmpobjOrder.push_back(objOrder[0]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[1]);
			tmpobjOrder.push_back(objOrder[2]);
			independentTrees.push_back(tmpobjOrder);
		}
		else if(scenePath.compare("/home/chaitanya/PoseDataset17/table/scene-0008/") == 0){
			std::vector<apc_objects::APCObjects*> tmpobjOrder;
			tmpobjOrder.push_back(objOrder[0]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[1]);
			tmpobjOrder.push_back(objOrder[2]);
			independentTrees.push_back(tmpobjOrder);
		}
		else if(scenePath.compare("/home/chaitanya/PoseDataset17/table/scene-0009/") == 0){
			std::vector<apc_objects::APCObjects*> tmpobjOrder;
			tmpobjOrder.push_back(objOrder[0]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[1]);
			tmpobjOrder.push_back(objOrder[2]);
			independentTrees.push_back(tmpobjOrder);
		}
		else if(scenePath.compare("/home/chaitanya/PoseDataset17/table/scene-0010/") == 0){
			std::vector<apc_objects::APCObjects*> tmpobjOrder;
			tmpobjOrder.push_back(objOrder[0]);
			tmpobjOrder.push_back(objOrder[1]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[2]);
			independentTrees.push_back(tmpobjOrder);
		}
		else if(scenePath.compare("/home/chaitanya/PoseDataset17/table/scene-0011/") == 0){
			std::vector<apc_objects::APCObjects*> tmpobjOrder;
			tmpobjOrder.push_back(objOrder[0]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[1]);
			tmpobjOrder.push_back(objOrder[2]);
			independentTrees.push_back(tmpobjOrder);
		}
		else if(scenePath.compare("/home/chaitanya/PoseDataset17/table/scene-0012/") == 0){
			std::vector<apc_objects::APCObjects*> tmpobjOrder;
			tmpobjOrder.push_back(objOrder[0]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[1]);
			tmpobjOrder.push_back(objOrder[2]);
			independentTrees.push_back(tmpobjOrder);
		}
		else if(scenePath.compare("/home/chaitanya/PoseDataset17/table/scene-0013/") == 0){
			std::vector<apc_objects::APCObjects*> tmpobjOrder;
			tmpobjOrder.push_back(objOrder[0]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[1]);
			tmpobjOrder.push_back(objOrder[2]);
			independentTrees.push_back(tmpobjOrder);
		}
		else if(scenePath.compare("/home/chaitanya/PoseDataset17/table/scene-0014/") == 0){
			std::vector<apc_objects::APCObjects*> tmpobjOrder;
			tmpobjOrder.push_back(objOrder[0]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[1]);
			tmpobjOrder.push_back(objOrder[2]);
			independentTrees.push_back(tmpobjOrder);
		}
		else if(scenePath.compare("/home/chaitanya/PoseDataset17/table/scene-0015/") == 0){
			std::vector<apc_objects::APCObjects*> tmpobjOrder;
			tmpobjOrder.push_back(objOrder[0]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[1]);
			tmpobjOrder.push_back(objOrder[2]);
			independentTrees.push_back(tmpobjOrder);
		}
		else if(scenePath.compare("/home/chaitanya/PoseDataset17/table/scene-0016/") == 0){
			std::vector<apc_objects::APCObjects*> tmpobjOrder;
			tmpobjOrder.push_back(objOrder[0]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[1]);
			tmpobjOrder.push_back(objOrder[2]);
			independentTrees.push_back(tmpobjOrder);
		}
		else if(scenePath.compare("/home/chaitanya/PoseDataset17/table/scene-0017/") == 0){
			std::vector<apc_objects::APCObjects*> tmpobjOrder;
			tmpobjOrder.push_back(objOrder[0]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[1]);
			tmpobjOrder.push_back(objOrder[2]);
			independentTrees.push_back(tmpobjOrder);
		}
		else if(scenePath.compare("/home/chaitanya/PoseDataset17/table/scene-0018/") == 0){
			std::vector<apc_objects::APCObjects*> tmpobjOrder;
			tmpobjOrder.push_back(objOrder[0]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[1]);
			tmpobjOrder.push_back(objOrder[2]);
			independentTrees.push_back(tmpobjOrder);
		}
		else if(scenePath.compare("/home/chaitanya/PoseDataset17/table/scene-0019/") == 0){
			std::vector<apc_objects::APCObjects*> tmpobjOrder;
			tmpobjOrder.push_back(objOrder[0]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[1]);
			tmpobjOrder.push_back(objOrder[2]);
			independentTrees.push_back(tmpobjOrder);
		}
		else if(scenePath.compare("/home/chaitanya/PoseDataset17/table/scene-0020/") == 0){
			std::vector<apc_objects::APCObjects*> tmpobjOrder;
			tmpobjOrder.push_back(objOrder[0]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[1]);
			tmpobjOrder.push_back(objOrder[2]);
			independentTrees.push_back(tmpobjOrder);
		}
		else if(scenePath.compare("/home/chaitanya/PoseDataset17/table/scene-0021/") == 0){
			std::vector<apc_objects::APCObjects*> tmpobjOrder;
			tmpobjOrder.push_back(objOrder[0]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[1]);
			tmpobjOrder.push_back(objOrder[2]);
			independentTrees.push_back(tmpobjOrder);
		}
		else if(scenePath.compare("/home/chaitanya/PoseDataset17/table/scene-0022/") == 0){
			std::vector<apc_objects::APCObjects*> tmpobjOrder;
			tmpobjOrder.push_back(objOrder[0]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[1]);
			tmpobjOrder.push_back(objOrder[2]);
			independentTrees.push_back(tmpobjOrder);
		}
		else if(scenePath.compare("/home/chaitanya/PoseDataset17/table/scene-0023/") == 0){
			std::vector<apc_objects::APCObjects*> tmpobjOrder;
			tmpobjOrder.push_back(objOrder[0]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[1]);
			tmpobjOrder.push_back(objOrder[2]);
			independentTrees.push_back(tmpobjOrder);
		}
		else if(scenePath.compare("/home/chaitanya/PoseDataset17/table/scene-0024/") == 0){
			std::vector<apc_objects::APCObjects*> tmpobjOrder;
			tmpobjOrder.push_back(objOrder[0]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[1]);
			tmpobjOrder.push_back(objOrder[2]);
			independentTrees.push_back(tmpobjOrder);
		}
		else if(scenePath.compare("/home/chaitanya/PoseDataset17/table/scene-0025/") == 0){
			std::vector<apc_objects::APCObjects*> tmpobjOrder;
			tmpobjOrder.push_back(objOrder[0]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[1]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[2]);
			independentTrees.push_back(tmpobjOrder);
		}
		else if(scenePath.compare("/home/chaitanya/PoseDataset17/table/scene-0026/") == 0){
			std::vector<apc_objects::APCObjects*> tmpobjOrder;
			tmpobjOrder.push_back(objOrder[0]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[1]);
			tmpobjOrder.push_back(objOrder[2]);
			independentTrees.push_back(tmpobjOrder);
		}
		else if(scenePath.compare("/home/chaitanya/PoseDataset17/table/scene-0027/") == 0){
			std::vector<apc_objects::APCObjects*> tmpobjOrder;
			tmpobjOrder.push_back(objOrder[0]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[1]);
			tmpobjOrder.push_back(objOrder[2]);
			independentTrees.push_back(tmpobjOrder);
		}
		else if(scenePath.compare("/home/chaitanya/PoseDataset17/table/scene-0028/") == 0){
			std::vector<apc_objects::APCObjects*> tmpobjOrder;
			tmpobjOrder.push_back(objOrder[0]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[1]);
			tmpobjOrder.push_back(objOrder[2]);
			independentTrees.push_back(tmpobjOrder);
		}
		else if(scenePath.compare("/home/chaitanya/PoseDataset17/table/scene-0029/") == 0){
			std::vector<apc_objects::APCObjects*> tmpobjOrder;
			tmpobjOrder.push_back(objOrder[0]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[1]);
			tmpobjOrder.push_back(objOrder[2]);
			independentTrees.push_back(tmpobjOrder);
		}
		else if(scenePath.compare("/home/chaitanya/PoseDataset17/table/scene-0030/") == 0){
			std::vector<apc_objects::APCObjects*> tmpobjOrder;
			tmpobjOrder.push_back(objOrder[0]);
			independentTrees.push_back(tmpobjOrder);

			tmpobjOrder.clear();
			tmpobjOrder.push_back(objOrder[1]);
			tmpobjOrder.push_back(objOrder[2]);
			independentTrees.push_back(tmpobjOrder);
		}
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

	/********************************* function: withinCLusterLookup ***************************************
	*******************************************************************************************************/
	void Scene::withinCLusterLookup(cv::Mat points, apc_objects::APCObjects* obj, Eigen::Matrix4f gtPose,
			 						cv::Mat clusterIndices, int bestClusterIdx){
		cv::Mat colMean, colMeanRepAll;

		float minRotErrWithinCluster_rot = INT_MAX;
		float minRotErrWithinCluster_trans = INT_MAX;

		float minTransErrWithinCluster_rot = INT_MAX;
		float minTransErrWithinCluster_trans = INT_MAX;

		float rotErr, transErr;
		for(int ii = 0; ii < points.rows; ii++){
			if(clusterIndices.at<int>(ii) == bestClusterIdx){
				Eigen::Matrix4f tmpPose;
				utilities::convert6DToMatrix(tmpPose, points, ii);
				utilities::getPoseError(tmpPose, gtPose, obj->symInfo, rotErr, transErr);
				if(rotErr < minRotErrWithinCluster_rot){
					minRotErrWithinCluster_rot = rotErr;
					minRotErrWithinCluster_trans = transErr;
				}
				if(transErr < minTransErrWithinCluster_trans){
					minTransErrWithinCluster_rot = rotErr;
					minTransErrWithinCluster_trans = transErr;
				}
			}
		}

		ofstream statsFile;
		statsFile.open ((scenePath + "debug/stats_" + obj->objName + ".txt").c_str(), std::ofstream::out | std::ofstream::app);
		statsFile << "minRotErrWithinCluster_rot: " << minRotErrWithinCluster_rot << std::endl;
		statsFile << "minRotErrWithinCluster_trans: " << minRotErrWithinCluster_trans << std::endl;
		statsFile << "minTransErrWithinCluster_rot: " << minTransErrWithinCluster_rot << std::endl;
		statsFile << "minTransErrWithinCluster_trans: " << minTransErrWithinCluster_trans << std::endl;
		statsFile.close();
	}

	/********************************* function: clusterRotWithinTrans *************************************
	*******************************************************************************************************/

	void Scene::clusterRotWithinTrans(std::vector<cv::Mat> &transClusters, std::vector<cv::Mat> &scoreTrans, cv::Mat& transCenters, int k,
									  apc_objects::APCObjects* obj, std::vector< std::pair <Eigen::Isometry3d, float> >& subsetPose, Eigen::Matrix4f gtPose){
		float minRotErrHierrCluster_rot = INT_MAX;
		float minRotErrHierrCluster_trans = INT_MAX;

		float minTransErrHierrCluster_rot = INT_MAX;
		float minTransErrHierrCluster_trans = INT_MAX;

		std::vector<cv::Mat> pointsClusters(k*k);
		std::vector<cv::Mat> scoreClusters(k*k);

		for(int transIter=0; transIter<k; transIter++){
			cv::Mat rotCenters, rotIndices;
			cv::Mat rotPts(transClusters[transIter].rows, 3, CV_32F);

			for(int dim=3;dim<6;dim++)
				transClusters[transIter].col(dim).copyTo(rotPts.col(dim-3));

			cv::kmeans(rotPts, /* num clusters */ k, rotIndices, cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 1000, 0.001)
			, /*int attempts*/ 10, cv::KMEANS_PP_CENTERS, rotCenters);

			for(int inClusterPt=0; inClusterPt<transClusters[transIter].rows; inClusterPt++){
				pointsClusters[transIter*k + rotIndices.at<int>(inClusterPt)].push_back(transClusters[transIter].row(inClusterPt));
				scoreClusters[transIter*k + rotIndices.at<int>(inClusterPt)].push_back(scoreTrans[transIter].row(inClusterPt));
			}

			// use best LCP score guy as the cluster representative
			cv::Mat bestLCPRepresentative;
			for(int rotIter=0; rotIter<k; rotIter++){
				cv::Point min_loc, max_loc;
				double min, max;
				cv::minMaxLoc(scoreClusters[transIter*k + rotIter], &min, &max, &min_loc, &max_loc);
				bestLCPRepresentative.push_back(pointsClusters[transIter*k + rotIter].row(max_loc.y));
			}

			// use the average translation from rotational clusters
			// std::vector<cv::Mat> avgTransRotClusters(k);
			// for(int kk=0; kk<k; kk++){
			// 	cv::reduce(pointsClusters[transIter*k + kk], avgTransRotClusters[kk], 0, CV_REDUCE_AVG);
			// }

			for(int jj=0;jj<k;jj++){
				cv::Mat clusterRep;
				float rotErr, transErr;
				Eigen::Matrix4f tmpPose;
				Eigen::Isometry3d hypPose;
				Eigen::Isometry3d isoPose;

				// use best LCP score guy as the cluster representative
				clusterRep.push_back(bestLCPRepresentative.row(jj));
				utilities::convert6DToMatrix(tmpPose, clusterRep, 0);

				// use the k-cluster centers from translational clustering
				// cv::hconcat(transCenters.row(transIter), rotCenters.row(jj),clusterRep);
				// utilities::convert6DToMatrix(tmpPose, clusterRep, 0);

				// use the average translation from rotational clusters
				// cv::hconcat(avgTransRotClusters[jj].colRange(0, 2), rotCenters.row(jj),clusterRep);
				// utilities::convert6DToMatrix(tmpPose, clusterRep, 0); 
				
				// perform ICP on the hypothesis
				utilities::convertToCamera(tmpPose, camPose);
				utilities::convertToIsometry3d(tmpPose, isoPose);
				state::State* tmpState = new state::State(0);
				tmpState->updateStateId(-3);
				tmpState->numObjects = 1;
				tmpState->updateNewObject(obj, std::make_pair(isoPose, 0.f), tmpState->numObjects);
				tmpState->performTrICP(scenePath, 0.9);
				utilities::convertToMatrix(tmpState->objects[tmpState->numObjects-1].second, tmpPose);
				utilities::convertToWorld(tmpPose, camPose);

				// compute the error from ground truth
				utilities::getPoseError(tmpPose, gtPose, obj->symInfo, rotErr, transErr);
				
				// add the cluster representative to the hypothesis set
				utilities::convertToCamera(tmpPose, camPose);
				utilities::convertToIsometry3d(tmpPose, hypPose);
				subsetPose.push_back(std::make_pair(hypPose, 0));

				if(rotErr < minRotErrHierrCluster_rot){
					minRotErrHierrCluster_rot = rotErr;
					minRotErrHierrCluster_trans = transErr;
				}
				if(transErr < minTransErrHierrCluster_trans){
					minTransErrHierrCluster_rot = rotErr;
					minTransErrHierrCluster_trans = transErr;
				}
			}
		}
		ofstream statsFile;
		statsFile.open ((scenePath + "debug/stats_" + obj->objName + ".txt").c_str(), std::ofstream::out | std::ofstream::app);
		statsFile << "minRotErrHierrCluster_rot: " << minRotErrHierrCluster_rot << std::endl;
		statsFile << "minRotErrHierrCluster_trans: " << minRotErrHierrCluster_trans << std::endl;
		statsFile << "minTransErrHierrCluster_rot: " << minTransErrHierrCluster_rot << std::endl;
		statsFile << "minTransErrHierrCluster_trans: " << minTransErrHierrCluster_trans << std::endl;
		statsFile.close();

		clusters.push_back(pointsClusters);
		clusterScores.push_back(scoreClusters);
	}

	/********************************* function: clusterTransPoseSet ***************************************
	*******************************************************************************************************/

	void Scene::clusterTransPoseSet(cv::Mat points, cv::Mat scores, std::vector<cv::Mat> &transClusters, std::vector<cv::Mat> &scoreTrans,
										 cv::Mat& transCenters, apc_objects::APCObjects* obj, int k){
		cv::Mat transPts(points.rows, 3, CV_32F);
		for(int dim=0;dim<3;dim++)
			points.col(dim).copyTo(transPts.col(dim));

		cv::Mat transIndices;
		cv::kmeans(transPts, /* num clusters */ k, transIndices, cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 1000, 0.001)
			, /*int attempts*/ 10, cv::KMEANS_PP_CENTERS, transCenters);

		for(int ii=0; ii<points.rows; ii++){
			transClusters[transIndices.at<int>(ii)].push_back(points.row(ii));
			scoreTrans[transIndices.at<int>(ii)].push_back(scores.row(ii));
		}
	}

	/********************************* function: clusterPoseSet *******************************************
	*******************************************************************************************************/
	void Scene::clusterPoseSet(cv::Mat points, cv::Mat &clusterIndices, cv::Mat &clusterCenters,
								int &bestClusterIdx, apc_objects::APCObjects* obj, Eigen::Matrix4f gtPose,
			 					int k, std::vector< std::pair <Eigen::Isometry3d, float> >& subsetPose){
		cv::Mat colMean, colMeanRepAll, colMeanRepK;
		cv::reduce(points, colMean, 0, CV_REDUCE_AVG);
		cv::repeat(colMean, points.rows, 1, colMeanRepAll);
		cv::repeat(colMean, k, 1, colMeanRepK);
		points = points/colMeanRepAll;

		cv::kmeans(points, /* num clusters */ k, clusterIndices, cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 1000, 0.001)
			, /*int attempts*/ 10, cv::KMEANS_PP_CENTERS, clusterCenters);

		float rotErr, transErr;
		float minRotErrCluster_rot = INT_MAX;
		float minRotErrCluster_trans = INT_MAX;

		float minTransErrCluster_rot = INT_MAX;
		float minTransErrCluster_trans = INT_MAX;

		bestClusterIdx = -1;
		clusterCenters = clusterCenters.mul(colMeanRepK);
		for(int ii = 0; ii < k; ii++){
			Eigen::Matrix4f tmpPose;
			Eigen::Isometry3d hypPose;
			utilities::convert6DToMatrix(tmpPose, clusterCenters, ii);

			Eigen::Isometry3d isoPose;
			utilities::convertToCamera(tmpPose, camPose);
			utilities::convertToIsometry3d(tmpPose, isoPose);
			state::State* tmpState = new state::State(0);
			tmpState->updateStateId(-4);
			tmpState->numObjects = 1;
			tmpState->updateNewObject(obj, std::make_pair(isoPose, 0.f), tmpState->numObjects);
			tmpState->performTrICP(scenePath, 0.9);
			utilities::convertToMatrix(tmpState->objects[tmpState->numObjects-1].second, tmpPose);
			utilities::convertToWorld(tmpPose, camPose);

			utilities::getPoseError(tmpPose, gtPose, obj->symInfo, rotErr, transErr);

			utilities::convertToCamera(tmpPose, camPose);
			utilities::convertToIsometry3d(tmpPose, hypPose);
			subsetPose.push_back(std::make_pair(hypPose, 0));

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
		points = points.mul(colMeanRepAll);

		ofstream statsFile;
		statsFile.open ((scenePath + "debug/stats_" + obj->objName + ".txt").c_str(), std::ofstream::out | std::ofstream::app);
		statsFile << "ClusterHypBestRotation_RotErr: " << minRotErrCluster_rot << std::endl;
		statsFile << "ClusterHypBestRotation_TransErr: " << minRotErrCluster_trans << std::endl;
		statsFile << "ClusterHypBestTranslation_RotErr: " << minTransErrCluster_rot << std::endl;
		statsFile << "ClusterHypBestTranslation_TransErr: " << minTransErrCluster_trans << std::endl;
		statsFile.close();
	}

	/********************************* function: customClustering ******************************************
	*******************************************************************************************************/
	void Scene::customClustering(cv::Mat points, cv::Mat &clusterIndices, cv::Mat &clusterCenters,
								int &bestClusterIdx, apc_objects::APCObjects* obj, Eigen::Matrix4f gtPose,
			 					int k, std::vector< std::pair <Eigen::Isometry3d, float> >& subsetPose){
		cv::Mat colMean, colMeanRepAll, colMeanRepK;
		cv::reduce(points, colMean, 0, CV_REDUCE_AVG);
		cv::repeat(colMean, points.rows, 1, colMeanRepAll);
		cv::repeat(colMean, k, 1, colMeanRepK);
		points = points/colMeanRepAll;

		cv::Mat centers(k,6,CV_32F);
		centers.setTo(0);
		cvflann::KMeansIndexParams k_params(50, 1000, cvflann::FLANN_CENTERS_KMEANSPP,0.001);
		int numCenters = cv::flann::hierarchicalClustering<cv::flann::L2<float> >(points, centers, k_params);

		std::cout << "number of clusters are: " << numCenters << std::endl;
		for(int i=0; i<centers.rows; i++){
			std::cout << centers.row(i) << std::endl;
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
		cutOffScore.push_back(lcpThreshold*bestscore);
		
		std::cout << "object hypothesis count: " << allPose.size() << std::endl;
		std::cout << "bestScore: " << bestscore <<std::endl;
		std::cout << "Scene::getHypothesis: Super4PCS time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
		std::cout << "###################################################" <<std::endl;

	#ifdef DBG_SUPER4PCS
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

		// check all hypothesis from super4pcs
		float rotErr, transErr, emdErr;
		float minRotErr_rot = INT_MAX;
		float minRotErr_trans = INT_MAX;

		float minTransErr_rot = INT_MAX;
		float minTransErr_trans = INT_MAX;

		float minEMDErr_rot = INT_MAX;
		float minEMDErr_trans = INT_MAX;
		float minEMDErr_emd = INT_MAX;

		cv::Mat points(allPose.size(), 6, CV_32F);
		cv::Mat scores(allPose.size(), 1, CV_32F);
		int bestGTPoseIdx = -1;
		int bestEMDPoseIdx = -1;
		const clock_t begin_time_s = clock();
		for(int ii = 0; ii < allPose.size(); ii++){
			Eigen::Matrix4f pose;
			utilities::convertToMatrix(allPose[ii].first, pose);
			utilities::convertToWorld(pose, camPose);
			utilities::addToCVMat(pose, points, ii);
			utilities::getPoseError(pose, gtPose, obj->symInfo, rotErr, transErr);
			
			if(obj->symInfo(0) == 360 && obj->symInfo(1) == 360 && obj->symInfo(2) == 360){
				emdErr = transErr;
			} else {
			utilities::getEMDError(pose, gtPose, obj->pclModelSparse, emdErr, 
				x_min - 0.05, x_max + 0.05, y_min - 0.05, y_max + 0.05, z_min - 0.05, z_max + 0.05);
			}

			scores.at<float>(ii, 0) = allPose[ii].second;

			if(rotErr < minRotErr_rot){
				minRotErr_rot = rotErr;
				minRotErr_trans = transErr;
				bestGTPoseIdx = ii;
			}
			if(transErr < minTransErr_trans){
				minTransErr_rot = rotErr;
				minTransErr_trans = transErr;
			}
			if(emdErr < minEMDErr_emd){
				minEMDErr_rot = rotErr;
				minEMDErr_trans = transErr;
				minEMDErr_emd = emdErr;
				bestEMDPoseIdx = ii;
			}
			// std::cout << "iteration: " << ii << " error: " <<  emdErr <<std::endl;
		}
		std::cout << "all hypothesis time: " << float( clock () - begin_time_s ) /  CLOCKS_PER_SEC << std::endl;
		ofstream statsFile;
		statsFile.open ((scenePath + "debug/stats_" + obj->objName + ".txt").c_str(), std::ofstream::out | std::ofstream::app);
		statsFile << "allHypBestRotation_RotErr: " << minRotErr_rot << std::endl;
		statsFile << "allHypBestRotation_TransErr: " << minRotErr_trans << std::endl;
		statsFile << "allHypBestTranslation_RotErr: " << minTransErr_rot << std::endl;
		statsFile << "allHypBestTranslation_TransErr: " << minTransErr_trans << std::endl;
		statsFile << "bestEMD_RotErr: " << minEMDErr_rot << std::endl;
		statsFile << "bestEMD_TransErr: " << minEMDErr_trans << std::endl;
		statsFile << "bestEMD_emdErr: " << minEMDErr_emd << std::endl;
		statsFile.close();

		std::vector< std::pair <Eigen::Isometry3d, float> > subsetPose;
		subsetPose.push_back(allPose[bestEMDPoseIdx]);
		subsetPose.push_back(allPose[bestGTPoseIdx]);

		// k-means clustering of poses
		// cv::Mat clusterIndices, clusterCenters;
		// int bestClusterIdx;
		// int k1 = 10;
		// clusterPoseSet(points, clusterIndices, clusterCenters, bestClusterIdx, obj, gtPose, k1, subsetPose);
		// withinCLusterLookup(points, obj, gtPose, clusterIndices, bestClusterIdx);

		// custom clustering of poses
		// cv::Mat clusterIndices, clusterCenters;
		// int bestClusterIdx;
		// int k1 = 50;
		// customClustering(points, clusterIndices, clusterCenters, bestClusterIdx, obj, gtPose, k1, subsetPose);
		// withinCLusterLookup(points, obj, gtPose, clusterIndices, bestClusterIdx);

		// hierarchical k-means clustering of poses: first translation, then rotation
		// cv::Mat transCenters;
		// int k2 = 5;
		// std::vector<cv::Mat> transClusters(k2);
		// std::vector<cv::Mat> scoreTrans(k2);
		// clusterTransPoseSet(points, scores, transClusters, scoreTrans, transCenters, obj, k2);
		// clusterRotWithinTrans(transClusters, scoreTrans, transCenters, k2, obj, subsetPose, gtPose);

		// add the poses to hypothesis set
		unconditionedHypothesis.push_back(subsetPose);
	#else
		unconditionedHypothesis.push_back(allPose);
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

		finalState->updateStateId(-1);

		physim::PhySim *tmpSim = new physim::PhySim();
		tmpSim->addTable(0.55);
		for(int ii=0; ii<objOrder.size(); ii++)
			tmpSim->initRigidBody(objOrder[ii]->objName);

		for(int i=0;i<objOrder.size();i++){
			Eigen::Matrix4f bestPoseMat;
			float rotErr, transErr;
			ofstream statsFile;

			finalState->numObjects = i+1;
			finalState->updateNewObject(objOrder[i], std::make_pair(max4PCSPose[i].first, 0.f), finalState->numObjects);
			utilities::convertToMatrix(max4PCSPose[i].first, bestPoseMat);
			utilities::convertToWorld(bestPoseMat, camPose);
			utilities::getPoseError(bestPoseMat, groundTruth[i].second, objOrder[i]->symInfo, rotErr, transErr);
			statsFile.open ((scenePath + "debug/stats_" + objOrder[i]->objName + ".txt").c_str(), std::ofstream::out | std::ofstream::app);
			statsFile << "BestLCP_RotErr: " << rotErr << std::endl;
			statsFile << "BestLCP_TransErr: " << transErr << std::endl;
			statsFile.close();

			// perform ICP and evaluate error
			finalState->performTrICP(scenePath, 0.9);
			utilities::convertToMatrix(finalState->objects[finalState->numObjects-1].second, bestPoseMat);
			utilities::convertToWorld(bestPoseMat, camPose);
			utilities::getPoseError(bestPoseMat, groundTruth[i].second, objOrder[i]->symInfo, rotErr, transErr);
			statsFile.open ((scenePath + "debug/stats_" + objOrder[i]->objName + ".txt").c_str(), std::ofstream::out | std::ofstream::app);
			statsFile << "BestLCP_AfterICP_RotErr: " << rotErr << std::endl;
			statsFile << "BestLCP_AfterICP_TransErr: " << transErr << std::endl;
			statsFile.close();

			// perform simulation and evaluate error
			// finalState->correctPhysics(tmpSim, camPose, scenePath);
			// utilities::convertToMatrix(finalState->objects[finalState->numObjects-1].second, bestPoseMat);
			// utilities::convertToWorld(bestPoseMat, camPose);
			// utilities::getPoseError(bestPoseMat, groundTruth[i].second, objOrder[i]->symInfo, rotErr, transErr);
			// statsFile.open ((scenePath + "debug/stats_" + objOrder[i]->objName + ".txt").c_str(), std::ofstream::out | std::ofstream::app);
			// statsFile << "BestLCP_AfterICPSimulation_RotErr: " << rotErr << std::endl;
			// statsFile << "BestLCP_AfterICPSimulation_TransErr: " << transErr << std::endl;
			// statsFile.close();

		}
		cv::Mat depth_image_minLCP;
		finalState->render(camPose, scenePath, depth_image_minLCP);
		finalState->computeCost(depth_image_minLCP, depthImage);

		// render closest to ground truth pose
 		state::State* bestStateEMD = new state::State(numObjects);
 		bestStateEMD->updateStateId(-8);
 		for(int i=0;i<objOrder.size();i++)
 			bestStateEMD->updateNewObject(objOrder[i], std::make_pair(unconditionedHypothesis[i][0].first, 0.f), numObjects);
 		
 		cv::Mat depth_image_minGT;
 		bestStateEMD->render(camPose, scenePath, depth_image_minGT);
 		bestStateEMD->computeCost(depth_image_minGT, depthImage);
 		
 		state::State* bestStateRot = new state::State(numObjects);
 		bestStateRot->updateStateId(-9);
 		for(int i=0;i<objOrder.size();i++)
 			bestStateRot->updateNewObject(objOrder[i], std::make_pair(unconditionedHypothesis[i][1].first, 0.f), numObjects);
 		cv::Mat depth_image_minGT2;
 		bestStateRot->render(camPose, scenePath, depth_image_minGT2);

		delete tmpSim;


	#else
		for(int i=0;i<objOrder.size();i++){
			std::vector< std::pair <Eigen::Isometry3d, float> > allPose;
			getHypothesis(objOrder[i], objOrder[i]->pclSegment, objOrder[i]->pclModel, allPose);
		}
	#endif
	}

	/********************************* end of functions ****************************************************
	*******************************************************************************************************/
} // namespace scene