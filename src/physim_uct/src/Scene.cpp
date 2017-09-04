#include <Scene.hpp>
#include <fstream>

#include <detection_package/UpdateActiveListFrame.h>
#include <detection_package/UpdateBbox.h>

#include <opencv2/flann/flann.hpp>

// global parameters
float clusteringLCPThreshold = 0.2;
float searchLCPThreshold = 0;
int clusteringTransDiscretization = 100;
int clusteringRotDiscretization = 2;
int k_clusters = 5;
int kkmeansIters = 3;

clock_t preprocess_begin_time;

// Super4PCS package
int getProbableTransformsSuper4PCS(std::string input1, std::string input2, Eigen::Isometry3d &bestPose, 
              float &bestscore, std::vector< std::pair <Eigen::Isometry3d, float> > &allPose);

namespace scene{

	/********************************* function: constructor ***********************************************
	*******************************************************************************************************/

	Scene::Scene(std::string scenePath){
		srand (time(NULL));
		
		preprocess_begin_time = clock();

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

	/********************************* function: kernelKMeans **********************************************
	*******************************************************************************************************/

	void Scene::kernelKMeans(cv::Mat &rotPts, cv::Mat &rotCenters, Eigen::Vector3f symInfo){
		std::vector<std::vector<int> > clusters (k_clusters, std::vector<int> (0));
		std::vector<int> rotCenterIndices(k_clusters);
		int num_pts = rotPts.rows;

		std::map<std::string, int> rotMap;
		for(int ptIdx_1=0; ptIdx_1<num_pts; ptIdx_1++){
			int r = int(rotPts.at<float>(ptIdx_1, 0)) - (int(rotPts.at<float>(ptIdx_1, 0))%clusteringRotDiscretization);
			int p = int(rotPts.at<float>(ptIdx_1, 1)) - (int(rotPts.at<float>(ptIdx_1, 1))%clusteringRotDiscretization);
			int y = int(rotPts.at<float>(ptIdx_1, 2)) - (int(rotPts.at<float>(ptIdx_1, 2))%clusteringRotDiscretization);

			char buf[50];
			sprintf(buf,"#%d#%d#%d#",r,p,y);
			std::string key(buf);
			rotMap[key] = 1;
		}
		std::cout << "Scene::kernelKMeans: trans cluster size: " << rotMap.size() << '\n';

		num_pts = rotMap.size();
		// check if the number of points are greater than the number of cluster center
		if(num_pts > k_clusters){

			cv::Mat points(num_pts, 3, CV_32F);
			int ptIdx=0;
			for(std::map<std::string,int>::iterator it=rotMap.begin(); it!=rotMap.end(); ++it){
	    		int roll,pitch,yaw;
	  			sscanf(it->first.c_str(),"#%d#%d#%d#",&roll,&pitch,&yaw);
				points.at<float>(ptIdx,0) = roll;
				points.at<float>(ptIdx,1) = pitch;
				points.at<float>(ptIdx,2) = yaw;
				ptIdx++;
			}

			// precompute rotation matrices and their transpose
			std::vector<Eigen::Matrix3f > rotMats, rotMatsTrans;
			for(int ptIdx_1=0; ptIdx_1<num_pts; ptIdx_1++){
				Eigen::Quaternionf q;
				Eigen::Vector3f rotXYZ;
				rotXYZ << points.at<float>(ptIdx_1, 0) * M_PI/180.0, 
							points.at<float>(ptIdx_1, 1) * M_PI/180.0,
							points.at<float>(ptIdx_1, 2) * M_PI/180.0;
				utilities::toQuaternion(rotXYZ, q);
				Eigen::Matrix3f rotm = q.toRotationMatrix();
				rotMats.push_back(rotm);
				rotMatsTrans.push_back(rotm.transpose());
			}

			// compute and store distances
			std::vector<std::vector<float> > lookupTable(num_pts, std::vector<float> (num_pts));
			for(int ptIdx_1=0; ptIdx_1<num_pts; ptIdx_1++)
				for(int ptIdx_2=ptIdx_1+1; ptIdx_2<num_pts; ptIdx_2++)
					lookupTable[ptIdx_1][ptIdx_2] = utilities::getRotDistance(rotMatsTrans[ptIdx_1], rotMats[ptIdx_2], symInfo);

			// assign random cluster centers
			for(int clusterIdx=0; clusterIdx<k_clusters; clusterIdx++)
				rotCenterIndices[clusterIdx] = rand() % num_pts;

			// perform iterations
			for(int iterIdx=0; iterIdx<kkmeansIters; iterIdx++) {

				// clear cluster assignments
				for(int clusterIdx=0; clusterIdx<k_clusters; clusterIdx++){
					clusters[clusterIdx].clear();
				}

				// assign points to clusters with closest cluster center
				for(int ptIdx=0; ptIdx<num_pts; ptIdx++) {
					float minRotDist = INT_MAX;
					int closestCluster = -1;
					for(int clusterIdx=0; clusterIdx<k_clusters; clusterIdx++){
						float rotDist = lookupTable[std::min(ptIdx, rotCenterIndices[clusterIdx])]
													[std::max(ptIdx, rotCenterIndices[clusterIdx])];
						if(rotDist < minRotDist){
							minRotDist = rotDist;
							closestCluster = clusterIdx;
						}
					}
					clusters[closestCluster].push_back(ptIdx);
				}

				// compute cluster representative
				for(int clusterIdx=0; clusterIdx<k_clusters; clusterIdx++){
					int repIndex = -1;
					float minDist = INT_MAX;
					int clusterSize = clusters[clusterIdx].size();

					if(clusterSize > 0){
						for(int outPts=0; outPts<clusterSize; outPts++){
							float dist = 0;

							for(int inPts=0; inPts<clusterSize; inPts++)
								dist += lookupTable[std::min(clusters[clusterIdx][outPts], clusters[clusterIdx][inPts])]
											[std::max(clusters[clusterIdx][outPts], clusters[clusterIdx][inPts])];

							if(dist < minDist){
								minDist = dist;
								repIndex = outPts;
							}
						}
						rotCenterIndices[clusterIdx] = clusters[clusterIdx][repIndex];
					}
					else
						rotCenterIndices[clusterIdx] = rand() % num_pts;
				}


			} // iteration ends

			for (int clusterIdx=0; clusterIdx<k_clusters; clusterIdx++)
				rotCenters.push_back(points.row(rotCenterIndices[clusterIdx]));
		}
		else {
			for(std::map<std::string,int>::iterator it=rotMap.begin(); it!=rotMap.end(); ++it){
				cv::Mat points(1, 3, CV_32F);
	    		int roll,pitch,yaw;
	  			sscanf(it->first.c_str(),"#%d#%d#%d#",&roll,&pitch,&yaw);
				points.at<float>(0,0) = roll;
				points.at<float>(0,1) = pitch;
				points.at<float>(0,2) = yaw;
				rotCenters.push_back(points.row(0));
			}
		}
	}

	/********************************* function: clusterRotWithinTrans *************************************
	*******************************************************************************************************/

	void Scene::clusterRotWithinTrans(std::vector<cv::Mat> &transClusters, std::vector<cv::Mat> &scoreTrans, cv::Mat& transCenters, 
								apc_objects::APCObjects* obj, std::vector< std::pair <Eigen::Isometry3d, float> >& subsetPose){

		for(int transIter=0; transIter<k_clusters; transIter++){
			cv::Mat rotCenters, rotIndices;
			cv::Mat rotPts(transClusters[transIter].rows, 3, CV_32F);

			for(int dim=3; dim<6; dim++)
				transClusters[transIter].col(dim).copyTo(rotPts.col(dim-3));

			kernelKMeans(rotPts, rotCenters, obj->symInfo);

			for(int rotIter=0; rotIter<k_clusters; rotIter++){
				cv::Mat clusterRep;
				Eigen::Matrix4f tmpPose;
				Eigen::Isometry3d hypPose;
				Eigen::Isometry3d isoPose;

				// use the k-cluster centers from translational clustering
				cv::hconcat(transCenters.row(transIter), rotCenters.row(rotIter),clusterRep);
				
				utilities::convert6DToMatrix(tmpPose, clusterRep, 0);
				utilities::convertToWorld(tmpPose, camPose);
				utilities::writePoseToFile(tmpPose, obj->objName, scenePath, "clusterPose");
				utilities::convertToCamera(tmpPose, camPose);
				utilities::convertToIsometry3d(tmpPose, hypPose);
				subsetPose.push_back(std::make_pair(hypPose, 0));
			}
		}
	}

	/********************************* function: clusterTransPoseSet ***************************************
	*******************************************************************************************************/

	void Scene::clusterTransPoseSet(cv::Mat points, cv::Mat scores, std::vector<cv::Mat> &transClusters, std::vector<cv::Mat> &scoreTrans,
										 cv::Mat& transCenters, apc_objects::APCObjects* obj){
		cv::Mat transPts(points.rows, 3, CV_32F);

		// get first 3 columns of 6DoF pose
		for(int dim=0;dim<3;dim++)
			points.col(dim).copyTo(transPts.col(dim));

		cv::Mat transIndices;
		cv::kmeans(transPts, k_clusters, transIndices, cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 1000, 0.001)
			, 10, cv::KMEANS_PP_CENTERS, transCenters);

		for(int ii=0; ii<points.rows; ii++){
			transClusters[transIndices.at<int>(ii)].push_back(points.row(ii));
			scoreTrans[transIndices.at<int>(ii)].push_back(scores.row(ii));
		}
	}

	/********************************* function: clusterPoseSet *******************************************
	*******************************************************************************************************/
	void Scene::clusterPoseSet(cv::Mat points, cv::Mat &clusterIndices, cv::Mat &clusterCenters, apc_objects::APCObjects* obj,
									 std::vector< std::pair <Eigen::Isometry3d, float> >& subsetPose){

		cv::Mat colMean, colMeanRepAll, colMeanRepK;
		cv::reduce(points, colMean, 0, CV_REDUCE_AVG);
		cv::repeat(colMean, points.rows, 1, colMeanRepAll);
		cv::repeat(colMean, k_clusters, 1, colMeanRepK);
		points = points/colMeanRepAll;

		cv::kmeans(points, k_clusters, clusterIndices, cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 1000, 0.001)
			, /*int attempts*/ 10, cv::KMEANS_PP_CENTERS, clusterCenters);

		clusterCenters = clusterCenters.mul(colMeanRepK);
		for(int ii = 0; ii < k_clusters; ii++){
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
			utilities::convertToIsometry3d(tmpPose, hypPose);
			subsetPose.push_back(std::make_pair(hypPose, 0));
		}
		points = points.mul(colMeanRepAll);
	}

	/********************************* function: getHypothesis *********************************************
	*******************************************************************************************************/
	void Scene::getHypothesis(apc_objects::APCObjects* obj, PointCloud::Ptr pclSegment, PointCloud::Ptr pclModel){
		const clock_t begin_time = clock();
		std::vector< std::pair <Eigen::Isometry3d, float> > subsetPose;

		std::string input1 = scenePath + "debug/pclSegment_" + obj->objName + ".ply";
		std::string input2 = scenePath + "debug/pclModel_" + obj->objName + ".ply";
		pcl::io::savePLYFile(input1, *pclSegment);
		pcl::io::savePLYFile(input2, *pclModel);

		Eigen::Isometry3d bestPose;
		float bestscore = 0;
		std::vector< std::pair <Eigen::Isometry3d, float> > superPCSposes;
		getProbableTransformsSuper4PCS(input1, input2, bestPose, bestscore, superPCSposes);
		max4PCSPose.push_back(std::make_pair(bestPose, bestscore));
		cutOffScore.push_back(searchLCPThreshold*bestscore);
		subsetPose.push_back(std::make_pair(bestPose, bestscore));
		
		std::cout << "Scene::getHypothesis: object hypothesis count: " << superPCSposes.size() << std::endl;
		std::cout << "Scene::getHypothesis: bestScore: " << bestscore <<std::endl;
		std::cout << "Scene::getHypothesis: super4PCS time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;

		std::map<std::string, float> poseMap;
		for(int ii = 0; ii < superPCSposes.size(); ii++) {
			Eigen::Matrix4f pose;
			utilities::convertToMatrix(superPCSposes[ii].first, pose);
			
			utilities::convertToWorld(pose, camPose);
			utilities::writePoseToFile(pose, obj->objName, scenePath, "allPose");
			utilities::convertToCamera(pose, camPose);

			cv::Mat cvPose(1,6, CV_32F);
			utilities::convertToCVMat(pose, cvPose);

			if(superPCSposes[ii].second > clusteringLCPThreshold*bestscore){
				char buf[50];
				int x = int(cvPose.at<float>(0, 0)*clusteringTransDiscretization);
				int y = int(cvPose.at<float>(0, 1)*clusteringTransDiscretization);
				int z = int(cvPose.at<float>(0, 2)*clusteringTransDiscretization);
				int roll = int(cvPose.at<float>(0, 3)) - (int(cvPose.at<float>(0, 3)) % clusteringRotDiscretization);
				int pitch = int(cvPose.at<float>(0, 4)) - (int(cvPose.at<float>(0, 4)) % clusteringRotDiscretization);
				int yaw = int(cvPose.at<float>(0, 5)) - (int(cvPose.at<float>(0, 5)) % clusteringRotDiscretization);

				sprintf(buf,"#%d#%d#%d#%d#%d#%d#", x, y, z, roll, pitch, yaw);
				std::string key(buf);
				poseMap[key] = std::max(poseMap[key], superPCSposes[ii].second);
			}
		}

		cv::Mat points(poseMap.size(), 6, CV_32F);
		cv::Mat scores(poseMap.size(), 1, CV_32F);
		int ptIdx=0;
		for(std::map<std::string,float>::iterator it=poseMap.begin(); it!=poseMap.end(); ++it){
    		int x,y,z,roll,pitch,yaw;
  			sscanf(it->first.c_str(),"#%d#%d#%d#%d#%d#%d#",&x,&y,&z,&roll,&pitch,&yaw);
			points.at<float>(ptIdx,0) = (float)x/clusteringTransDiscretization;
			points.at<float>(ptIdx,1) = (float)y/clusteringTransDiscretization;
			points.at<float>(ptIdx,2) = (float)z/clusteringTransDiscretization;
			points.at<float>(ptIdx,3) = roll;
			points.at<float>(ptIdx,4) = pitch;
			points.at<float>(ptIdx,5) = yaw;
			scores.at<float>(ptIdx, 0) = it->second;
			ptIdx++;
		}
		std::cout << "Scene::getHypothesis: hypothesis set after discretization: " << poseMap.size() << '\n';

		// k-means clustering of poses
		// cv::Mat clusterIndices, clusterCenters;
		// clusterPoseSet(points, clusterIndices, clusterCenters, obj, subsetPose);

		// hierarchical k-means clustering of poses: first translation, then rotation
		cv::Mat transCenters;
		std::vector<cv::Mat> transClusters(k_clusters);
		std::vector<cv::Mat> scoreTrans(k_clusters);
		clusterTransPoseSet(points, scores, transClusters, scoreTrans, transCenters, obj);
		clusterRotWithinTrans(transClusters, scoreTrans, transCenters, obj, subsetPose);

		unconditionedHypothesis.push_back(subsetPose);
	}

	/********************************* function: getUnconditionedHypothesis ********************************
	*******************************************************************************************************/

	void Scene::getUnconditionedHypothesis(){

		for(int i=0;i<objOrder.size();i++){
			std::cout << std::endl;
			std::cout << "*****Scene::getHypothesis Begins*****" << std::endl;

			getHypothesis(objOrder[i], objOrder[i]->pclSegment, objOrder[i]->pclModel);

			std::cout << "*****Scene::getHypothesis Ends*******" << std::endl;
			std::cout << std::endl;
		}

		finalState->updateStateId(-1);
		for(int i=0;i<objOrder.size();i++){
			Eigen::Matrix4f bestPoseMat;

			finalState->numObjects = i+1;
			finalState->updateNewObject(objOrder[i], std::make_pair(max4PCSPose[i].first, 0.f), finalState->numObjects);
			utilities::convertToMatrix(max4PCSPose[i].first, bestPoseMat);
			utilities::convertToWorld(bestPoseMat, camPose);
			utilities::writePoseToFile(bestPoseMat, finalState->objects[finalState->numObjects-1].first->objName, scenePath, "super4pcs");

			finalState->performTrICP(scenePath, 0.9);

			utilities::convertToMatrix(finalState->objects[finalState->numObjects-1].second, bestPoseMat);
			utilities::convertToWorld(bestPoseMat, camPose);
		    utilities::writePoseToFile(bestPoseMat, finalState->objects[finalState->numObjects-1].first->objName, scenePath, "super4pcs");
		}
		cv::Mat depth_image_minLCP;
		finalState->render(camPose, scenePath, depth_image_minLCP);
		finalState->computeCost(depth_image_minLCP, depthImage);

		#ifdef DBG_SUPER4PCS
		    ofstream pFile;
			pFile.open ((scenePath + "debug/scores.txt").c_str(), std::ofstream::out | std::ofstream::app);
			pFile << finalState->score << " " << (float( clock () - preprocess_begin_time ) /  CLOCKS_PER_SEC) << std::endl;
			pFile.close();
		#endif
	}

	/********************************* end of functions ****************************************************
	*******************************************************************************************************/
} // namespace scene