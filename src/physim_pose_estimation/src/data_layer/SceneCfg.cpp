#include <SceneCfg.hpp>
#include <Segmentation.hpp>
#include <HypothesisSelection.hpp>
#include <fstream>

clock_t preprocess_begin_time;

namespace scene_cfg{

	/********************************* function: constructor ***********************************************
	*******************************************************************************************************/

	SceneCfg::SceneCfg(std::string SceneFiles, std::string SegmentationMode, 
						std::string HypothesisGenerationMode, std::string HypothesisVerificationMode){
		srand (time(NULL));
		
		preprocess_begin_time = clock();

		scenePath = SceneFiles;
		segMode = SegmentationMode;
		hypoGenMode = HypothesisGenerationMode;
		HVMode = HypothesisVerificationMode;
	}

	/********************************* function: destructor ************************************************
	*******************************************************************************************************/

	SceneCfg::~SceneCfg(){
	}

	/********************************* function: removeTable ***********************************************
	References: http://pointclouds.org/documentation/tutorials/planar_segmentation.php,
	http://pointclouds.org/documentation/tutorials/extract_indices.php
	*******************************************************************************************************/

	void SceneCfg::removeTable(){
		PointCloudRGB::Ptr SampledSceneCloud(new PointCloudRGB);
		sceneCloud = PointCloudRGB::Ptr(new PointCloudRGB);
		utilities::convert3dOrganizedRGB(depthImage, colorImage, camIntrinsic, sceneCloud);

		// Creating the filtering object: downsample the dataset using a leaf size of 0.5cm
		pcl::VoxelGrid<pcl::PointXYZRGB> sor;
		sor.setInputCloud (sceneCloud);
		sor.setLeafSize (0.005f, 0.005f, 0.005f);
		sor.filter (*SampledSceneCloud);

		pcl::io::savePLYFile(scenePath + "debug_super4PCS/scene.ply", *SampledSceneCloud);

		// Plane Fitting
		pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
		pcl::SACSegmentation<pcl::PointXYZRGB> seg;
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
				pcl::PointXYZRGB pt;
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

	void SceneCfg::getTableParams(){

		PointCloudRGB::Ptr SampledSceneCloud(new PointCloudRGB);
		PointCloudRGB::Ptr SampledTableCloud(new PointCloudRGB);
		PointCloudRGB::Ptr tableCloudTrans (new PointCloudRGB);
		PointCloudRGB::Ptr sceneCloudTrans (new PointCloudRGB);

		pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
		pcl::SACSegmentation<pcl::PointXYZRGB> seg;

		PointCloudRGB::Ptr cloud_p(new PointCloudRGB);
		pcl::ExtractIndices<pcl::PointXYZRGB> extract;

		pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
		PointCloudRGB Final;

		float mean_z = 0;
		Eigen::Matrix4f tablePose;
		Eigen::Matrix4f icpTransform;

		pcl::io::loadPLYFile(scenePath + "debug_super4PCS/scene.ply", *SampledSceneCloud);
		pcl::io::loadPLYFile(scenePath + "table.ply", *SampledTableCloud);

		// get table inlier points
		pcl::transformPointCloud(*SampledSceneCloud, *sceneCloudTrans, camPose);
		seg.setOptimizeCoefficients (true);
		seg.setModelType (pcl::SACMODEL_PLANE);
		seg.setMethodType (pcl::SAC_MSAC);
		seg.setDistanceThreshold (0.005);
		seg.setMaxIterations (1000);
		seg.setInputCloud (sceneCloudTrans);
		seg.segment (*inliers, *coefficients);
		extract.setInputCloud (sceneCloudTrans);
	    extract.setIndices (inliers);
	    extract.setNegative (false);
	    extract.filter (*cloud_p);

	    // get estimated table pose
		for(int ii=0; ii<cloud_p->points.size(); ii++)
			mean_z += cloud_p->points[ii].z;
		mean_z /= cloud_p->points.size();
		tablePose.setIdentity();
		tablePose(0,3) = 0.7;
		tablePose(2,3) = mean_z - 0.2/*table height*/; 
		pcl::transformPointCloud(*SampledTableCloud, *tableCloudTrans, tablePose);

		// performing icp
		icp.setInputSource(sceneCloudTrans);
		icp.setInputTarget(tableCloudTrans);
		icp.setMaxCorrespondenceDistance (0.01);
		icp.setMaximumIterations(50);
		icp.setTransformationEpsilon (1e-9);
		icp.align(Final);
		icpTransform = icp.getFinalTransformation();
		tablePose = icpTransform.inverse().eval()*tablePose;
		
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

	/********************************* getSceneInfo ********************************************************
	*******************************************************************************************************/

	void APCSceneCfg::getSceneInfo(GlobalCfg *gCfg){
		std::vector<double> camPose7D;
		XmlRpc::XmlRpcValue camIntr;

		// Loading params from the yaml file
		system(("rosparam load " + scenePath + "gt_info.yml").c_str());

		gCfg->nh.getParam("/camera/camera_pose", camPose7D);
		gCfg->nh.getParam("/scene/num_objects", numObjects);

		camPose = Eigen::Matrix4f::Zero(4,4);
		utilities::toTransformationMatrix(camPose, camPose7D);

		// Loading RGB and depth images
		colorImage = cv::imread(scenePath + "frame-000000.color.png", CV_LOAD_IMAGE_COLOR);
	    utilities::readDepthImage(depthImage, scenePath + "frame-000000.depth.png");

		// Loading scene objects
		for(int ii=0; ii<numObjects; ii++){
			std::string currObject;
			char objTopic[50];
			sprintf(objTopic, "/scene/object_%d/name", ii+1);
			gCfg->nh.getParam(objTopic, currObject);

			for(int jj=0; jj< gCfg->num_objects; jj++){\
				if(!currObject.compare(gCfg->gObjects[jj]->objName)) {

					std::cout << "Loading object: " << currObject << std::endl;

					scene_cfg::SceneObjects *tSceneObj = new scene_cfg::SceneObjects();
					tSceneObj->pObject = gCfg->gObjects[jj];
					tSceneObj->objMask = cv::Mat::zeros(colorImage.rows, colorImage.cols, CV_32FC1);
					tSceneObj->objPose.matrix().setIdentity();
					pSceneObjects.push_back(tSceneObj);
				}
			}
		}

	  	// Reading camera intrinsic matrix
		camIntrinsic = Eigen::Matrix3f::Zero(3,3);
		gCfg->nh.getParam("/camera/camera_intrinsics", camIntr);
		for(int32_t ii = 0; ii < camIntr.size(); ii++)
			for(int32_t jj = 0; jj < camIntr[ii].size(); jj++)
				camIntrinsic(ii, jj) = static_cast<double>(camIntr[ii][jj]);
	}

	void YCBSceneCfg::getSceneInfo(GlobalCfg *gCfg){
		std::vector<double> camPose7D;
		XmlRpc::XmlRpcValue camIntr;

		// Loading params from the yaml file
		system(("rosparam load " + scenePath + "gt_info.yml").c_str());

		gCfg->nh.getParam("/camera/camera_pose", camPose7D);
		gCfg->nh.getParam("/scene/num_objects", numObjects);

		camPose = Eigen::Matrix4f::Zero(4,4);
		utilities::toTransformationMatrix(camPose, camPose7D);

		// Loading RGB and depth images
		colorImage = cv::imread(scenePath + "frame-000000.color.png", CV_LOAD_IMAGE_COLOR);
	    utilities::readDepthImage(depthImage, scenePath + "frame-000000.depth.png");

		// Loading scene objects
		for(int ii=0; ii<numObjects; ii++){
			std::string currObject;
			char objTopic[50];
			sprintf(objTopic, "/scene/object_%d/name", ii+1);
			gCfg->nh.getParam(objTopic, currObject);

			for(int jj=0; jj< gCfg->num_objects; jj++){\
				if(!currObject.compare(gCfg->gObjects[jj]->objName)) {

					std::cout << "Loading object: " << currObject << std::endl;

					scene_cfg::SceneObjects *tSceneObj = new scene_cfg::SceneObjects();
					tSceneObj->pObject = gCfg->gObjects[jj];
					tSceneObj->objMask = cv::Mat::zeros(colorImage.rows, colorImage.cols, CV_32FC1);
					tSceneObj->objPose.matrix().setIdentity();
					pSceneObjects.push_back(tSceneObj);
				}
			}
		}

	  	// Reading camera intrinsic matrix
		camIntrinsic = Eigen::Matrix3f::Zero(3,3);
		gCfg->nh.getParam("/camera/camera_intrinsics", camIntr);
		for(int32_t ii = 0; ii < camIntr.size(); ii++)
			for(int32_t jj = 0; jj < camIntr[ii].size(); jj++)
				camIntrinsic(ii, jj) = static_cast<double>(camIntr[ii][jj]);
	}
	
	/********************************* cleanDebugLocations *************************************************
	*******************************************************************************************************/

	void APCSceneCfg::cleanDebugLocations(){
		system(("rm -rf " + scenePath + "debug_search").c_str());
		system(("mkdir " + scenePath + "debug_search").c_str());

		system(("rm -rf " + scenePath + "debug_super4PCS").c_str());
		system(("mkdir " + scenePath + "debug_super4PCS").c_str());

		system(("rm " + scenePath + "result.txt").c_str());
	}

	void YCBSceneCfg::cleanDebugLocations(){
		system(("rm -rf " + scenePath + "debug_search").c_str());
		system(("mkdir " + scenePath + "debug_search").c_str());

		system(("rm -rf " + scenePath + "debug_super4PCS").c_str());
		system(("mkdir " + scenePath + "debug_super4PCS").c_str());

		system(("rm " + scenePath + "result.txt").c_str());
	}
	
	/********************************* perfromSegmentation *************************************************
	*******************************************************************************************************/

	void SceneCfg::perfromSegmentation(GlobalCfg *pCfg){
		// if nothing is specified, it uses the ground truth segmentation
		segmentation::Segmentation *pSegmentation;
		if(!segMode.compare("RCNN"))
			pSegmentation = new segmentation::RCNNSegmentation();
		else if(!segMode.compare("FCN"))
			pSegmentation = new segmentation::FCNSegmentation();
		else if(!segMode.compare("FCNThreshold"))
			pSegmentation = new segmentation::FCNThresholdSegmentation();
		else
			pSegmentation = new segmentation::GTSegmentation();

		pSegmentation->compute2dSegment(pCfg, this);
		pSegmentation->compute3dSegment(this);
	}

	/********************************* generateHypothesis **************************************************
	*******************************************************************************************************/

	void SceneCfg::generateHypothesis(){
		for(int ii=0; ii<numObjects; ii++){
			pSceneObjects[ii]->hypotheses = new pose_candidates::ObjectPoseCandidateSet();
			pSceneObjects[ii]->hypotheses->generate(pSceneObjects[ii]->pObject->objName, scenePath, pSceneObjects[ii]->pclSegment, pSceneObjects[ii]->pObject->pclModel, camIntrinsic);
		}
	}
	/********************************* performHypothesisSelection ******************************************
	*******************************************************************************************************/

	void SceneCfg::performHypothesisSelection(){
		hypothesis_selection::HypothesisSelection *hSelect;

		if(!HVMode.compare("LCP"))
			hSelect = new hypothesis_selection::LCPSelection();
		else
			hSelect = new hypothesis_selection::LCPSelection();

		hSelect->selectBestPoses(this);
	}
	/********************************* end of functions ****************************************************
	*******************************************************************************************************/
} // namespace scene