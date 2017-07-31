#include <Scene.hpp>
#include <fstream>

#include <detection_package/UpdateActiveListFrame.h>
#include <detection_package/UpdateBbox.h>

namespace scene{
	Scene::Scene(std::string scenePath){
		// Read objects file and add the scene objects
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

		// Read camera pose
		camPose = Eigen::Matrix4f::Zero(4,4);
		filein.open ((scenePath + "cameraExtrinsic.txt").c_str(), std::ifstream::in);
		for(int i=0; i<4; i++)
			for(int j=0; j<4; j++)
	  			filein >> camPose(i,j);
	  	filein.close();

	  	// Read camera intrinsic matrix
		camIntrinsic = Eigen::Matrix3f::Zero(3,3);
		filein.open ((scenePath + "cameraIntinsic.txt").c_str(), std::ifstream::in);
		for(int i=0; i<3; i++)
			for(int j=0; j<3; j++)
	  			filein >> camIntrinsic(i,j);
	  	filein.close();

	  	colorImage = cv::imread(scenePath + "frame-000000.color.png", CV_LOAD_IMAGE_COLOR);
	    utilities::readDepthImage(depthImage, scenePath + "frame-000000.depth.png");
	} // function: constructor

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
    			ROS_INFO("Returned : %d", (bool)listsrv.response.result);
  		  	else{
    			ROS_ERROR("Failed to call service UpdateActiveListFrame");
    			exit(1);
    	  	}
    	}

    	// Call R-CNN
    	boxsrv.request.scene_path = scenePath;
    	if (clientbox.call(boxsrv))
      		for(int i=0;i<numObjects;i++)
        		sceneObjs[i]->bbox	= cv::Rect(boxsrv.response.tl_x[i], boxsrv.response.tl_y[i], boxsrv.response.br_x[i] - boxsrv.response.tl_x[i],
                    							boxsrv.response.br_y[i] - boxsrv.response.tl_y[i]);
        else{
      		ROS_ERROR("Failed to call service UpdateBbox");
      		exit(1);
    	}
	} // function: performRCNNDetection

	void Scene::get3DSegments(){
		for(int i=0;i<numObjects;i++){
			std::cout<<sceneObjs[i]->objName<<std::endl;
			cv::Mat mask = cv::Mat::zeros(colorImage.rows, colorImage.cols, CV_32FC1);
			mask(sceneObjs[i]->bbox) = 1;
			cv::Mat objDepth = depthImage.mul(mask);
			sceneObjs[i]->pclSegment = PointCloud::Ptr(new PointCloud);
			utilities::convert3d(objDepth, camIntrinsic, sceneObjs[i]->pclSegment);

			// boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
			// viewer = utilities::simpleVis(sceneObjs[i]->pclSegment);
			// while (!viewer->wasStopped ()){
			// 	viewer->spinOnce (100);
			// 	boost::this_thread::sleep (boost::posix_time::microseconds (100000));
			// }
		}
	} // function: get3DSegments

	// reference: http://pointclouds.org/documentation/tutorials/planar_segmentation.php,
	// http://pointclouds.org/documentation/tutorials/extract_indices.php
	void Scene::removeTable(){
		PointCloud::Ptr SampledSceneCloud(new PointCloud);
		sceneCloud = PointCloud::Ptr(new PointCloud);
		utilities::convert3d(depthImage, camIntrinsic, sceneCloud);
		
		// Create the filtering object: downsample the dataset using a leaf size of 0.5cm
		pcl::VoxelGrid<pcl::PointXYZ> sor;
		sor.setInputCloud (sceneCloud);
		sor.setLeafSize (0.005f, 0.005f, 0.005f);
		sor.filter (*SampledSceneCloud);

		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
		viewer = utilities::simpleVis(SampledSceneCloud);
		while (!viewer->wasStopped ()){
			viewer->spinOnce (100);
			boost::this_thread::sleep (boost::posix_time::microseconds (100000));
		}

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

		//Classify table, non-table points
		PointCloud::Ptr tableCloud(new PointCloud);
		PointCloud::Ptr objectPoints(new PointCloud);
		for(int i=0;i<sceneCloud->points.size();i++){
			pcl::PointXYZ pt = sceneCloud->points[i];
			double dist = pcl::pointToPlaneDistance(pt, coefficients->values[0], coefficients->values[1],
														coefficients->values[2], coefficients->values[3]);
			if(dist<0.005 && sceneCloud->points[i].z)
				tableCloud->points.push_back(pt);
			else if(sceneCloud->points[i].z)
				objectPoints->points.push_back(pt);
		}

		depthImage = cv::Mat::zeros(depthImage.rows, depthImage.cols, CV_32FC1);
		utilities::convert2d(depthImage, camIntrinsic, objectPoints);
		utilities::writeDepthImage(depthImage, scenePath+"/debug/scene.png");

		viewer = utilities::simpleVis(objectPoints);
		while (!viewer->wasStopped ()){
			viewer->spinOnce (100);
			boost::this_thread::sleep (boost::posix_time::microseconds (100000));
		}

	} // function: removeTable

	std::vector<apc_objects::APCObjects*> Scene::getOrder(){
		return sceneObjs;
	} // function: getOrder

} // namespace scene