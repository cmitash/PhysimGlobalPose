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
		system(("rosparam load " + scenePath + "gt_info.yml").c_str());

		ros::NodeHandle nh;

		// Reading camera pose
		camPose = Eigen::Matrix4f::Zero(4,4);
		std::vector<double> camPose7D;
		nh.getParam("/camera/camera_pose", camPose7D);
		utilities::toTransformationMatrix(camPose, camPose7D);

		nh.getParam("/scene/num_objects", numObjects);
		for(int ii=0; ii<numObjects; ii++){
			std::string currObject;
			char objTopic[50];
			sprintf(objTopic, "/scene/object_%d/name", ii+1);
			nh.getParam(objTopic, currObject);
			for(int i=0;i<Objects.size();i++){
				if(!currObject.compare(Objects[i]->objName))
					sceneObjs.push_back(Objects[i]);
			}
		}
		this->scenePath = scenePath;

	  	// Reading camera intrinsic matrix
		camIntrinsic = Eigen::Matrix3f::Zero(3,3);
	  	XmlRpc::XmlRpcValue camIntr;
		nh.getParam("/camera/camera_intrinsics", camIntr);
		for(int32_t ii = 0; ii < camIntr.size(); ii++)
			for(int32_t jj = 0; jj < camIntr[ii].size(); jj++)
				camIntrinsic(ii, jj) = static_cast<double>(camIntr[ii][jj]);

	  	finalState = new state::State(0);
	  	colorImage = cv::imread(scenePath + "frame-000000.color.png", CV_LOAD_IMAGE_COLOR);
	    utilities::readDepthImage(depthImage, scenePath + "frame-000000.depth.png");
	}

	/********************************* function: destructor ************************************************
	*******************************************************************************************************/

	Scene::~Scene(){
		delete finalState;
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
        		sceneObjs[i]->bbox	= cv::Rect(boxsrv.response.tl_x[i], boxsrv.response.tl_y[i], 
        										boxsrv.response.br_x[i] - boxsrv.response.tl_x[i],
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

	/********************************* function: getGTSegments *********************************************
	*******************************************************************************************************/

	void Scene::getGTSegments(){
		for(int i=0;i<numObjects;i++){
			std::cout << "Scene::getGTSegments: " << sceneObjs[i]->objName << " index: "<< sceneObjs[i]->objIdx << std::endl;
			cv::Mat mask = cv::Mat::zeros(colorImage.rows, colorImage.cols, CV_32FC1);

			cv::Mat classImage = cv::imread(scenePath + "frame-000000.mask.png", -1);

			int imgWidth = classImage.cols;
			int imgHeight = classImage.rows;

			for(int u=0; u<imgHeight; u++)
				for(int v=0; v<imgWidth; v++){
					int classVal = (int)classImage.at<uchar>(u,v);
					if(sceneObjs[i]->objIdx == classVal){
						mask.at<float>(u,v) = 1.0;
					}
				}

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

	/********************************* function: getSegmentationPrior **************************************
	*******************************************************************************************************/

	void Scene::getSegmentationPrior(){
		cv::Mat priorSegImage = cv::Mat::zeros(colorImage.rows, colorImage.cols, CV_8UC1);
		cv::Mat currProbImage = cv::Mat::zeros(colorImage.rows, colorImage.cols, CV_32FC1);

		for(int i=0;i<numObjects;i++){
			std::cout << "Scene::getSegmentationPrior: " << sceneObjs[i]->objName << " index: "<< sceneObjs[i]->objIdx << std::endl;

			char imgName[50];
			cv::Mat probImage;
			sprintf(imgName, "frame-000000.segm.%s.png", sceneObjs[i]->objName.c_str());
			utilities::readDepthImage(probImage, (scenePath + imgName).c_str());
			int imgWidth = probImage.cols;
			int imgHeight = probImage.rows;

			for(int u=0; u<imgHeight; u++)
				for(int v=0; v<imgWidth; v++){
					float prob = probImage.at<float>(u,v);
					if(prob > currProbImage.at<float>(u,v) && depthImage.at<float>(u,v) > 0){
						priorSegImage.at<uchar>(u,v) = sceneObjs[i]->objIdx;
						currProbImage.at<float>(u,v) = prob;
					}
				}
		}
		cv::imwrite(scenePath + "segm.png", priorSegImage);
	}

	/********************************* function: removeTable ***********************************************
	References: http://pointclouds.org/documentation/tutorials/planar_segmentation.php,
	http://pointclouds.org/documentation/tutorials/extract_indices.php
	*******************************************************************************************************/

	void Scene::removeTable(){
		PointCloud::Ptr SampledSceneCloud(new PointCloud);
		sceneCloud = PointCloud::Ptr(new PointCloud);
		utilities::convert3dOrganized(depthImage, camIntrinsic, sceneCloud);

		std::string input1 = scenePath + "debug_super4PCS/scene.ply";
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
		utilities::writeDepthImage(depthImage, scenePath + "debug_search/scene.png");
	}

	/********************************* function: getTableParams ********************************************
	*******************************************************************************************************/

	void Scene::getTableParams(){
		cv::Mat tmpDepthImage;
		utilities::readDepthImage(tmpDepthImage, scenePath + "frame-000000.depth.png");
		PointCloud::Ptr SampledSceneCloud(new PointCloud);
		PointCloud::Ptr tmpsceneCloud(new PointCloud);
		utilities::convert3dOrganized(tmpDepthImage, camIntrinsic, tmpsceneCloud);
		
		// Creating the filtering object: downsample the dataset using a leaf size of 0.5cm
		pcl::VoxelGrid<pcl::PointXYZ> sor;
		sor.setInputCloud (tmpsceneCloud);
		sor.setLeafSize (0.005f, 0.005f, 0.005f);
		sor.filter (*SampledSceneCloud);

		PointCloud::Ptr sceneCloudTrans (new PointCloud);
		pcl::transformPointCloud(*SampledSceneCloud, *sceneCloudTrans, camPose);

		// Plane Fitting
		pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
		pcl::SACSegmentation<pcl::PointXYZ> seg;
		seg.setOptimizeCoefficients (true);
		seg.setModelType (pcl::SACMODEL_PLANE);
		seg.setMethodType (pcl::SAC_MSAC);
		seg.setDistanceThreshold (0.005);
		seg.setMaxIterations (1000);
		seg.setInputCloud (sceneCloudTrans);
		seg.segment (*inliers, *coefficients);

		PointCloud::Ptr cloud_p(new PointCloud);
		pcl::ExtractIndices<pcl::PointXYZ> extract;
		extract.setInputCloud (sceneCloudTrans);
	    extract.setIndices (inliers);
	    extract.setNegative (false);
	    extract.filter (*cloud_p);
		pcl::io::savePLYFile(scenePath + "inliers.ply", *cloud_p);

		float mean_z = 0;
		for(int ii=0;ii<cloud_p->points.size();ii++)
			mean_z += cloud_p->points[ii].z;
		mean_z /= cloud_p->points.size();

		PointCloud::Ptr tableCloud (new PointCloud);
		PointCloud::Ptr SampledTableCloud(new PointCloud);
		pcl::io::loadPLYFile("/home/chaitanya/Extended-Rutgers-RGBD-dataset/table_sampled.ply", *tableCloud);
		sor.setInputCloud (tableCloud);
		sor.setLeafSize (0.005f, 0.005f, 0.005f);
		sor.filter (*SampledTableCloud);

		Eigen::Matrix4f tablePose;
		tablePose.setIdentity();
		tablePose(0,3) = 0.7;
		tablePose(2,3) = mean_z - 0.2; /*table height*/
		PointCloud::Ptr tableCloudTrans (new PointCloud);
		pcl::transformPointCloud(*SampledTableCloud, *tableCloudTrans, tablePose);

		pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
		icp.setInputSource(sceneCloudTrans);
		icp.setInputTarget(tableCloudTrans);
		icp.setMaxCorrespondenceDistance (0.01);
		icp.setMaximumIterations(50);
		icp.setTransformationEpsilon (1e-9);
		pcl::PointCloud<pcl::PointXYZ> Final;
		icp.align(Final);
		Eigen::Matrix4f icpTransform = icp.getFinalTransformation();
		tablePose = icpTransform.inverse().eval()*tablePose;
		
		// pcl::io::savePLYFile(scenePath + "scene_before.ply", *sceneCloudTrans);
		// pcl::io::savePLYFile(scenePath + "scene_after.ply", Final);
		
		// pcl::io::savePLYFile(scenePath + "table_before.ply", *tableCloudTrans);
		pcl::transformPointCloud(*SampledTableCloud, *tableCloudTrans, tablePose);
		// pcl::io::savePLYFile(scenePath + "table_after.ply", *tableCloudTrans);

		tableParams.push_back(tablePose(0,0));
		tableParams.push_back(tablePose(0,1));
		tableParams.push_back(tablePose(0,2));
		tableParams.push_back(tablePose(0,3));
		tableParams.push_back(tablePose(1,0));
		tableParams.push_back(tablePose(1,1));
		tableParams.push_back(tablePose(1,2));
		tableParams.push_back(tablePose(1,3));
		tableParams.push_back(tablePose(2,0));
		tableParams.push_back(tablePose(2,1));
		tableParams.push_back(tablePose(2,2));
		tableParams.push_back(tablePose(2,3));
	}

	/********************************* function: getOrder **************************************************
	*******************************************************************************************************/

	void Scene::getOrder(){
		ros::NodeHandle nh;
		XmlRpc::XmlRpcValue my_list;
		nh.getParam("/scene/dependency_order", my_list);
		std::vector<apc_objects::APCObjects*> tmpobjOrder;
		for(int32_t ii = 0; ii < my_list.size(); ii++){
			tmpobjOrder.clear();
			for(int32_t jj = 0; jj < my_list[ii].size(); jj++){
				tmpobjOrder.push_back(sceneObjs[static_cast<int>(my_list[ii][jj])-1]);
				std::cout << "Scene::getOrder: " << sceneObjs[static_cast<int>(my_list[ii][jj])-1]->objName << std::endl;
			}
			std::cout << "Scene::getOrder New Tree" << std::endl; 
			independentTrees.push_back(tmpobjOrder);
		}
	}

	/********************************* function: kernelKMeans **********************************************
	*******************************************************************************************************/

	void Scene::kernelKMeans(cv::Mat &rotPoints, cv::Mat &rotScores, Eigen::Vector3f symInfo, cv::Mat &rotCenters,
		cv::Mat &rotCenterScores){

		std::vector<std::vector<int> > clusters (k_clusters, std::vector<int> (0));
		std::vector<int> rotCenterIndices(k_clusters);
		int num_pts = rotPoints.rows;

		std::map<std::string, float> rotMap;
		for(int ptIdx_1=0; ptIdx_1<num_pts; ptIdx_1++){
			int r = int(rotPoints.at<float>(ptIdx_1, 0)) - (int(rotPoints.at<float>(ptIdx_1, 0))%clusteringRotDiscretization);
			int p = int(rotPoints.at<float>(ptIdx_1, 1)) - (int(rotPoints.at<float>(ptIdx_1, 1))%clusteringRotDiscretization);
			int y = int(rotPoints.at<float>(ptIdx_1, 2)) - (int(rotPoints.at<float>(ptIdx_1, 2))%clusteringRotDiscretization);

			char buf[50];
			sprintf(buf,"#%d#%d#%d#",r,p,y);
			std::string key(buf);
			rotMap[key] = std::max(rotMap[key], rotScores.at<float>(ptIdx_1));
		}
		std::cout << "Scene::kernelKMeans: trans cluster size: " << rotMap.size() << '\n';

		num_pts = rotMap.size();

		// check if the number of points are greater than the number of cluster center
		if(num_pts > k_clusters){

			cv::Mat points(num_pts, 3, CV_32F);
			cv::Mat scores(num_pts, 1, CV_32F);

			int ptIdx=0;
			for(std::map<std::string, float>::iterator it=rotMap.begin(); it!=rotMap.end(); ++it){
	    		int roll,pitch,yaw;
	  			sscanf(it->first.c_str(),"#%d#%d#%d#",&roll,&pitch,&yaw);
				points.at<float>(ptIdx,0) = roll;
				points.at<float>(ptIdx,1) = pitch;
				points.at<float>(ptIdx,2) = yaw;
				scores.at<float>(ptIdx) = it->second;
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

			for (int clusterIdx=0; clusterIdx<k_clusters; clusterIdx++){
				rotCenters.push_back(points.row(rotCenterIndices[clusterIdx]));
				rotCenterScores.push_back(scores.row(rotCenterIndices[clusterIdx]));
			}
		}
		else {
			for(std::map<std::string,float>::iterator it=rotMap.begin(); it!=rotMap.end(); ++it){
				cv::Mat points(1, 3, CV_32F);
				cv::Mat scores(1, 1, CV_32F);
	    		int roll,pitch,yaw;

	  			sscanf(it->first.c_str(),"#%d#%d#%d#",&roll,&pitch,&yaw);
				points.at<float>(0,0) = roll;
				points.at<float>(0,1) = pitch;
				points.at<float>(0,2) = yaw;
				scores.at<float>(0) = it->second;
				rotCenters.push_back(points.row(0));
				rotCenterScores.push_back(scores.row(0));
			}
		}
	}

	/********************************* function: performKernelKMeansRotation *******************************
	*******************************************************************************************************/

	void Scene::performKernelKMeansRotation(std::vector<cv::Mat> &transClusters, std::vector<cv::Mat> &transScores, 
		cv::Mat& transCenters, apc_objects::APCObjects* obj, cv::Mat &clusterReps, cv::Mat &allClusterScores){

		for(int transIter=0; transIter<k_clusters; transIter++){
			cv::Mat rotCenters, rotIndices, rotCenterScores;
			cv::Mat rotPoints(transClusters[transIter].rows, 3, CV_32F);

			for(int dim=3; dim<6; dim++)
				transClusters[transIter].col(dim).copyTo(rotPoints.col(dim-3));

			kernelKMeans(rotPoints, transScores[transIter], obj->symInfo, rotCenters, rotCenterScores);

			for(int rotIter=0; rotIter<k_clusters; rotIter++){
				cv::Mat clusterRep;
				cv::hconcat(transCenters.row(transIter), rotCenters.row(rotIter), clusterRep);
				clusterReps.push_back(clusterRep);
				allClusterScores.push_back(rotCenterScores.row(rotIter));
			}
		}
	}

	/********************************* function: performKMeansTranslation **********************************
	*******************************************************************************************************/

	void Scene::performKMeansTranslation(cv::Mat points, cv::Mat scores, std::vector<cv::Mat> &transClusters, 
		std::vector<cv::Mat> &scoreTrans, cv::Mat& transCenters, apc_objects::APCObjects* obj){

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

	/********************************* function: performKMeans *********************************************
	*******************************************************************************************************/
	void Scene::performKMeans(cv::Mat points, cv::Mat &clusterIndices, cv::Mat &clusterCenters, 
		apc_objects::APCObjects* obj, std::vector< std::pair <Eigen::Isometry3d, float> >& subsetPose){

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
	void Scene::getHypothesis(apc_objects::APCObjects* obj, std::pair <Eigen::Isometry3d, float> &bestLCPPose, 
		std::vector< std::pair <Eigen::Isometry3d, float> > &allSuperPCSposes){

		const clock_t begin_time = clock();
		float bestscore = 0;
		Eigen::Isometry3d bestPose;

		std::string input1 = scenePath + "debug_super4PCS/pclSegment_" + obj->objName + ".ply";
		std::string input2 = scenePath + "debug_super4PCS/pclModel_" + obj->objName + ".ply";

		// saving the segment cloud with normal
		pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> ne;
		ne.setInputCloud (obj->pclSegment);
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
		ne.setSearchMethod (tree);
		pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals_segment (new pcl::PointCloud<pcl::PointNormal>);
		copyPointCloud(*obj->pclSegment, *cloud_normals_segment);
		ne.setRadiusSearch (0.01);
		ne.compute (*cloud_normals_segment);

		ne.setInputCloud (obj->pclModel);
		ne.setSearchMethod (tree);
		pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals_model (new pcl::PointCloud<pcl::PointNormal>);
		copyPointCloud(*obj->pclModel, *cloud_normals_model);
		ne.setRadiusSearch (0.01);
		ne.compute (*cloud_normals_model);

		pcl::io::savePLYFile(input1, *cloud_normals_segment);
		pcl::io::savePLYFile(input2, *cloud_normals_model);

		// pcl::io::savePLYFile(input1, *obj->pclSegment);
		// pcl::io::savePLYFile(input2, *obj->pclModel);
		
		getProbableTransformsSuper4PCS(input1, input2, bestPose, bestscore, allSuperPCSposes);
		bestLCPPose = std::make_pair(bestPose, bestscore);
		max4PCSPose.push_back(bestLCPPose);
		cutOffScore.push_back(searchLCPThreshold*bestscore);
		
		std::cout << "Scene::getHypothesis: object hypothesis count: " << allSuperPCSposes.size() << std::endl;
		std::cout << "Scene::getHypothesis: bestScore: " << bestscore <<std::endl;
		std::cout << "Scene::getHypothesis: super4PCS time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
	}

	/********************************* function: descretizeHypothesisSet ***********************************
	*******************************************************************************************************/

	void Scene::descretizeHypothesisSet(apc_objects::APCObjects* obj, float bestscore, std::map<std::string, float> &poseMap, 
		std::vector< std::pair <Eigen::Isometry3d, float> > &allSuperPCSposes){

		for(int ii = 0; ii < allSuperPCSposes.size(); ii++) {
			Eigen::Matrix4f pose;
			utilities::convertToMatrix(allSuperPCSposes[ii].first, pose);
			
			utilities::convertToWorld(pose, camPose);
			utilities::writePoseToFile(pose, obj->objName, scenePath, "debug_super4PCS/allPose");
			utilities::writeScoreToFile(allSuperPCSposes[ii].second, obj->objName, scenePath, "debug_super4PCS/allScore");
			utilities::convertToCamera(pose, camPose);

			cv::Mat cvPose(1,6, CV_32F);
			utilities::convertToCVMat(pose, cvPose);

			if(allSuperPCSposes[ii].second > clusteringLCPThreshold*bestscore){
				char buf[50];
				int x = int(cvPose.at<float>(0, 0)*clusteringTransDiscretization);
				int y = int(cvPose.at<float>(0, 1)*clusteringTransDiscretization);
				int z = int(cvPose.at<float>(0, 2)*clusteringTransDiscretization);
				int roll = int(cvPose.at<float>(0, 3)) - (int(cvPose.at<float>(0, 3)) % clusteringRotDiscretization);
				int pitch = int(cvPose.at<float>(0, 4)) - (int(cvPose.at<float>(0, 4)) % clusteringRotDiscretization);
				int yaw = int(cvPose.at<float>(0, 5)) - (int(cvPose.at<float>(0, 5)) % clusteringRotDiscretization);

				sprintf(buf,"#%d#%d#%d#%d#%d#%d#", x, y, z, roll, pitch, yaw);
				std::string key(buf);
				poseMap[key] = std::max(poseMap[key], allSuperPCSposes[ii].second);
			}
		}
	}

	/********************************* function: clusterHypothesisSet **************************************
	*******************************************************************************************************/

	void Scene::clusterHypothesisSet(apc_objects::APCObjects* obj, std::map<std::string, float> &poseMap,
		std::vector< std::pair <Eigen::Isometry3d, float> > &clusteredPoses){

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
		std::cout << "Scene::clusterHypothesisSet: hypothesis set after discretization: " << poseMap.size() << '\n';

		// k-means clustering of poses
		// cv::Mat clusterIndices, clusterCenters;
		// performKMeans(points, clusterIndices, clusterCenters, obj, subsetPose);

		// hierarchical k-means clustering of poses: first translation, then rotation
		cv::Mat transCenters;
		std::vector<cv::Mat> transClusters(k_clusters);
		std::vector<cv::Mat> transScores(k_clusters);
		cv::Mat allClusterReps;
		cv::Mat allClusterScores;

		performKMeansTranslation(points, scores, transClusters, transScores, transCenters, obj);
		performKernelKMeansRotation(transClusters, transScores, transCenters, obj, allClusterReps, allClusterScores);

		for(int clusterIter=0; clusterIter<k_clusters*k_clusters; clusterIter++){
			Eigen::Matrix4f tmpPose;
			Eigen::Isometry3d hypPose;
			
			utilities::convert6DToMatrix(tmpPose, allClusterReps, clusterIter);
			utilities::convertToWorld(tmpPose, camPose);
			utilities::writePoseToFile(tmpPose, obj->objName, scenePath, "debug_super4PCS/clusterPose");
			utilities::writeScoreToFile(allClusterScores.at<float>(clusterIter), obj->objName, scenePath, "debug_super4PCS/clusterScore");
			utilities::convertToCamera(tmpPose, camPose);
			utilities::convertToIsometry3d(tmpPose, hypPose);
			clusteredPoses.push_back(std::make_pair(hypPose, allClusterScores.at<float>(clusterIter)));
		}
	}

	/********************************* function: computeHypothesisSet **************************************
	*******************************************************************************************************/

	void Scene::computeHypothesisSet(){

		for(int i=0;i<numObjects;i++){
			std::cout << std::endl;
			std::cout << "*****Scene::getHypothesis Begins*****" << std::endl;

			std::map<std::string, float> poseMap;
			std::pair <Eigen::Isometry3d, float> bestLCPPose;
			std::vector< std::pair <Eigen::Isometry3d, float> > allSuperPCSposes;
			std::vector< std::pair <Eigen::Isometry3d, float> > clusteredPoses;
			std::vector< std::pair <Eigen::Isometry3d, float> > subsetPoses;

			getHypothesis(sceneObjs[i], bestLCPPose, allSuperPCSposes);
			descretizeHypothesisSet(sceneObjs[i], bestLCPPose.second, poseMap, allSuperPCSposes);
			clusterHypothesisSet(sceneObjs[i], poseMap, clusteredPoses);

			// Whether or not, you want to use the best LCP Pose
			clusteredPoses.push_back(bestLCPPose);
			
			unconditionedHypothesis.push_back(clusteredPoses);

			std::cout << "*****Scene::getHypothesis Ends*******" << std::endl;
			std::cout << std::endl;
		}

		finalState->updateStateId(-1);
		for(int i=0;i<sceneObjs.size();i++){
			Eigen::Matrix4f bestPoseMat;

			finalState->numObjects = i+1;
			finalState->updateNewObject(sceneObjs[i], std::make_pair(max4PCSPose[i].first, 0.f), finalState->numObjects);
			utilities::convertToMatrix(max4PCSPose[i].first, bestPoseMat);
			utilities::convertToWorld(bestPoseMat, camPose);
			utilities::writePoseToFile(bestPoseMat, finalState->objects[finalState->numObjects-1].first->objName, scenePath, "debug_super4PCS/super4pcs");

			finalState->performTrICP(scenePath, 0.9);
			finalState->render(camPose, scenePath);

			utilities::convertToMatrix(finalState->objects[finalState->numObjects-1].second, bestPoseMat);
			utilities::convertToWorld(bestPoseMat, camPose);
		    utilities::writePoseToFile(bestPoseMat, finalState->objects[finalState->numObjects-1].first->objName, scenePath, "debug_super4PCS/super4pcs");
		}
		finalState->computeCost(depthImage);
	}

	void Scene::readHypothesis(){
		ifstream pPoseFile, pScoreFile, pSuper4PCSFile;
		
		for(int i=0;i<sceneObjs.size();i++){
			Eigen::Matrix4f poseMat;
			float bestScore = 0;

			poseMat.setIdentity();
			pPoseFile.open ((scenePath +  "debug_super4PCS/clusterPose_" + sceneObjs[i]->objName + ".txt").c_str(), std::ofstream::in);
			pScoreFile.open ((scenePath +  "debug_super4PCS/clusterScore_" + sceneObjs[i]->objName + ".txt").c_str(), std::ofstream::in);

			std::vector< std::pair <Eigen::Isometry3d, float> > clusteredPoses;
			while(pPoseFile >> poseMat(0,0) >> poseMat(0,1) >> poseMat(0,2) >> poseMat(0,3) 
					 >> poseMat(1,0) >> poseMat(1,1) >> poseMat(1,2) >> poseMat(1,3)
					 >> poseMat(2,0) >> poseMat(2,1) >> poseMat(2,2) >> poseMat(2,3) ){
				Eigen::Isometry3d hypPose;
				float score;

				pScoreFile >> score;
				if(score > bestScore)bestScore = score;

				utilities::convertToCamera(poseMat, camPose);
				utilities::convertToIsometry3d(poseMat, hypPose);

				clusteredPoses.push_back(std::make_pair(hypPose, score));
			}

			// add the best super4PCS hypothesis as well
			Eigen::Isometry3d isoPose;
			Eigen::Matrix4f bestPoseMat;
			bestPoseMat.setIdentity();

			pSuper4PCSFile.open ((scenePath +  "debug_super4PCS/super4pcs_" + sceneObjs[i]->objName + ".txt").c_str(), std::ofstream::in);
			pSuper4PCSFile >> bestPoseMat(0,0) >> bestPoseMat(0,1) >> bestPoseMat(0,2) >> bestPoseMat(0,3) 
					 >> bestPoseMat(1,0) >> bestPoseMat(1,1) >> bestPoseMat(1,2) >> bestPoseMat(1,3)
					 >> bestPoseMat(2,0) >> bestPoseMat(2,1) >> bestPoseMat(2,2) >> bestPoseMat(2,3);

			pSuper4PCSFile >> bestPoseMat(0,0) >> bestPoseMat(0,1) >> bestPoseMat(0,2) >> bestPoseMat(0,3) 
					 >> bestPoseMat(1,0) >> bestPoseMat(1,1) >> bestPoseMat(1,2) >> bestPoseMat(1,3)
					 >> bestPoseMat(2,0) >> bestPoseMat(2,1) >> bestPoseMat(2,2) >> bestPoseMat(2,3);
			
			utilities::convertToCamera(bestPoseMat, camPose);
			utilities::convertToIsometry3d(bestPoseMat, isoPose);
			pSuper4PCSFile.close();
			clusteredPoses.push_back(std::make_pair(isoPose, bestScore));
			cutOffScore.push_back(searchLCPThreshold*bestScore);

			unconditionedHypothesis.push_back(clusteredPoses);
			pPoseFile.close();
			pScoreFile.close();
		}

		// create final state from super4PCS result
		finalState->updateStateId(-1);
		for(int i=0;i<sceneObjs.size();i++){
			Eigen::Isometry3d isoPose;
			Eigen::Matrix4f bestPoseMat;

			pSuper4PCSFile.open ((scenePath +  "debug_super4PCS/super4pcs_" + sceneObjs[i]->objName + ".txt").c_str(), std::ofstream::in);
			pSuper4PCSFile >> bestPoseMat(0,0) >> bestPoseMat(0,1) >> bestPoseMat(0,2) >> bestPoseMat(0,3) 
					 >> bestPoseMat(1,0) >> bestPoseMat(1,1) >> bestPoseMat(1,2) >> bestPoseMat(1,3)
					 >> bestPoseMat(2,0) >> bestPoseMat(2,1) >> bestPoseMat(2,2) >> bestPoseMat(2,3);

			pSuper4PCSFile >> bestPoseMat(0,0) >> bestPoseMat(0,1) >> bestPoseMat(0,2) >> bestPoseMat(0,3) 
					 >> bestPoseMat(1,0) >> bestPoseMat(1,1) >> bestPoseMat(1,2) >> bestPoseMat(1,3)
					 >> bestPoseMat(2,0) >> bestPoseMat(2,1) >> bestPoseMat(2,2) >> bestPoseMat(2,3);
			utilities::convertToCamera(bestPoseMat, camPose);
			utilities::convertToIsometry3d(bestPoseMat, isoPose);

			finalState->numObjects = i+1;
			finalState->updateNewObject(sceneObjs[i], std::make_pair(isoPose, 0.f), finalState->numObjects);
			finalState->render(camPose, scenePath);
			pSuper4PCSFile.close();
		}
		finalState->computeCost(depthImage);

	}
	/********************************* end of functions ****************************************************
	*******************************************************************************************************/
} // namespace scene