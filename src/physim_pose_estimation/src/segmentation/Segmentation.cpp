#include <Segmentation.hpp>

#include <rcnn_detection_package/UpdateActiveListFrame.h>
#include <rcnn_detection_package/UpdateBbox.h>

#include <fcn_segmentation_package/UpdateActiveListFrame.h>
#include <fcn_segmentation_package/UpdateSeg.h>

namespace segmentation{
  
  /********************************* function: constructor ***********************************************
  *******************************************************************************************************/
  Segmentation::Segmentation(){
  }

  /********************************* function: destructor ************************************************
  *******************************************************************************************************/
  Segmentation::~Segmentation(){
    
  }

	/********************************* function: compute2dSegment ******************************************
	*******************************************************************************************************/

	void RCNNSegmentation::compute2dSegment(GlobalCfg *gCfg, scene_cfg::SceneCfg *sCfg){
		ros::ServiceClient clientlist = gCfg->nh.serviceClient<rcnn_detection_package::UpdateActiveListFrame>("/update_active_list_and_frame");
		ros::ServiceClient clientbox = gCfg->nh.serviceClient<rcnn_detection_package::UpdateBbox>("/update_bbox");
		rcnn_detection_package::UpdateActiveListFrame listsrv;
		rcnn_detection_package::UpdateBbox boxsrv;

		// Update object list
		for(int ii=0; ii<sCfg->numObjects; ii++){
			listsrv.request.active_list.push_back(sCfg->pSceneObjects[ii]->pObject->objIdx);
			listsrv.request.active_frame = "000000";
		    if(clientlist.call(listsrv))
    			  ROS_INFO("Segmentation::compute2dSegment, object: %d", ii+1);
  		  	else{
      			ROS_ERROR("Failed to call service UpdateActiveListFrame");
      			exit(1);
    	  	}
    	}

    	// Calling R-CNN
    	boxsrv.request.scene_path = sCfg->scenePath;
    	if (clientbox.call(boxsrv)){
      		for(int ii=0; ii<sCfg->numObjects; ii++){
        		cv::Rect box = cv::Rect(boxsrv.response.tl_x[ii], boxsrv.response.tl_y[ii], 
        										        boxsrv.response.br_x[ii] - boxsrv.response.tl_x[ii],
                    				        boxsrv.response.br_y[ii] - boxsrv.response.tl_y[ii]);
            sCfg->pSceneObjects[ii]->objMask(box) = 1.0;
          }
      }
      else{
    		ROS_ERROR("Failed to call service UpdateBbox");
    		exit(1);
    	}
	}

  void RCNNThresholdSegmentation::compute2dSegment(GlobalCfg *gCfg, scene_cfg::SceneCfg *sCfg){
    ros::ServiceClient clientlist = gCfg->nh.serviceClient<rcnn_detection_package::UpdateActiveListFrame>("/update_active_list_and_frame");
    ros::ServiceClient clientbox = gCfg->nh.serviceClient<rcnn_detection_package::UpdateBbox>("/update_bbox");
    rcnn_detection_package::UpdateActiveListFrame listsrv;
    rcnn_detection_package::UpdateBbox boxsrv;

    // Update object list
    for(int ii=0; ii<sCfg->numObjects; ii++){
      listsrv.request.active_list.push_back(sCfg->pSceneObjects[ii]->pObject->objIdx);
      listsrv.request.active_frame = "000000";
        if(clientlist.call(listsrv))
            ROS_INFO("Segmentation::compute2dSegment, object: %d", ii+1);
          else{
            ROS_ERROR("Failed to call service UpdateActiveListFrame");
            exit(1);
          }
      }

      // Calling R-CNN
      boxsrv.request.scene_path = sCfg->scenePath;
      if (clientbox.call(boxsrv)){
          for(int ii=0; ii<sCfg->numObjects; ii++){
            cv::Rect box = cv::Rect(boxsrv.response.tl_x[ii], boxsrv.response.tl_y[ii], 
                                    boxsrv.response.br_x[ii] - boxsrv.response.tl_x[ii],
                                    boxsrv.response.br_y[ii] - boxsrv.response.tl_y[ii]);
            sCfg->pSceneObjects[ii]->objMask(box) = 1.0;
            cv::Mat probImage = cv::Mat::zeros(sCfg->pSceneObjects[ii]->objMask.rows, sCfg->pSceneObjects[ii]->objMask.cols, CV_16UC1);
            probImage(box) = 10000;
            cv::imwrite(sCfg->scenePath + "debug_super4PCS/" + sCfg->pSceneObjects[ii]->pObject->objName + ".png", probImage);
          }
      }
      else{
        ROS_ERROR("Failed to call service UpdateBbox");
        exit(1);
      }
  }

  void FCNSegmentation::compute2dSegment(GlobalCfg *gCfg, scene_cfg::SceneCfg *sCfg){
    ros::ServiceClient clientlist = gCfg->nh.serviceClient<fcn_segmentation_package::UpdateActiveListFrame>("/update_active_list_and_frame");
    ros::ServiceClient clientbox = gCfg->nh.serviceClient<fcn_segmentation_package::UpdateSeg>("/handle_get_segments");
    fcn_segmentation_package::UpdateActiveListFrame listsrv;
    fcn_segmentation_package::UpdateSeg segsrv;

    // Update object list
    for(int ii=0; ii<sCfg->numObjects; ii++){
      listsrv.request.active_list.push_back(sCfg->pSceneObjects[ii]->pObject->objIdx);
      listsrv.request.active_frame = "000000";
      if(clientlist.call(listsrv))
        ROS_INFO("Segmentation::compute2dSegment, object: %d", ii+1);
      else {
        ROS_ERROR("Failed to call service UpdateActiveListFrame");
        exit(1);
        }
      }


      // Calling FCN
      segsrv.request.scene_path = sCfg->scenePath;
      if (clientbox.call(segsrv)){
        // use the argmax prediction
        cv::Mat classImage = cv::imread(sCfg->scenePath + "debug_super4PCS/frame-000000.fcn.mask.png", -1);
        int imgWidth = classImage.cols;
        int imgHeight = classImage.rows;

        for(int u=0; u<imgHeight; u++){
          for(int v=0; v<imgWidth; v++) {
            int classVal = classImage.at<unsigned short>(u,v);
            for(int ii=0; ii<sCfg->numObjects; ii++){
              if(sCfg->pSceneObjects[ii]->pObject->objIdx == classVal)
                sCfg->pSceneObjects[ii]->objMask.at<float>(u,v) = 1.0;
            }
          }
        }

      }
      else{
        ROS_ERROR("Failed to call service UpdateSeg");
        exit(1);
      }
  }

  void FCNThresholdSegmentation::compute2dSegment(GlobalCfg *gCfg, scene_cfg::SceneCfg *sCfg){
    ros::ServiceClient clientlist = gCfg->nh.serviceClient<fcn_segmentation_package::UpdateActiveListFrame>("/update_active_list_and_frame");
    ros::ServiceClient clientbox = gCfg->nh.serviceClient<fcn_segmentation_package::UpdateSeg>("/handle_get_segments");
    fcn_segmentation_package::UpdateActiveListFrame listsrv;
    fcn_segmentation_package::UpdateSeg segsrv;

    // Update object list
    for(int ii=0; ii<sCfg->numObjects; ii++){
      listsrv.request.active_list.push_back(sCfg->pSceneObjects[ii]->pObject->objIdx);
      listsrv.request.active_frame = "000000";
      if(clientlist.call(listsrv))
        ROS_INFO("Segmentation::compute2dSegment, object: %d", ii+1);
      else {
        ROS_ERROR("Failed to call service UpdateActiveListFrame");
        exit(1);
      }
    }

      // Calling FCN
      segsrv.request.scene_path = sCfg->scenePath;
      if (clientbox.call(segsrv)){
        // cv::Mat sumImg = cv::Mat::zeros(480, 640, CV_32FC1);
        // for(int ii=0; ii<sCfg->numObjects; ii++){
        //   cv::Mat probImage;
        //   utilities::readDepthImage(probImage, sCfg->scenePath + "debug_super4PCS/" + sCfg->pSceneObjects[ii]->pObject->objName + ".png");
        //   cv::add(probImage, sumImg, sumImg);
        // }

        for(int ii=0; ii<sCfg->numObjects; ii++){
          cv::Mat probImage, bkgProbImg;
          utilities::readProbImage(probImage, sCfg->scenePath + "debug_super4PCS/" + sCfg->pSceneObjects[ii]->pObject->objName + ".png");
          utilities::readProbImage(bkgProbImg, sCfg->scenePath + "debug_super4PCS/background.png");
          int imgWidth = probImage.cols;
          int imgHeight = probImage.rows;

          for(int u=0; u<imgHeight; u++){
            for(int v=0; v<imgWidth; v++) {
              // if(!sumImg.at<float>(u,v))continue;
              float probVal = probImage.at<float>(u,v);
              float bkgProb = bkgProbImg.at<float>(u,v);
              // probVal = probVal/(float)sumImg.at<float>(u,v);
              if(probVal > 0.4 && bkgProb < 0.8)
                sCfg->pSceneObjects[ii]->objMask.at<float>(u,v) = 1.0;
              }
          }

        }
      }
      else{
        ROS_ERROR("Failed to call service UpdateSeg");
        exit(1);
      }
  }

  /********************************* function: compute2dSegment ******************************************
  *******************************************************************************************************/

  void GTSegmentation::compute2dSegment(GlobalCfg *gCfg, scene_cfg::SceneCfg *sCfg){
    for(int ii=0; ii<sCfg->numObjects; ii++){
      cv::Mat classImage = cv::imread(sCfg->scenePath + "frame-000000.mask.png", -1);
      cv::Mat probImage = cv::Mat::zeros(sCfg->pSceneObjects[ii]->objMask.rows, sCfg->pSceneObjects[ii]->objMask.cols, CV_16UC1);

      int imgWidth = classImage.cols;
      int imgHeight = classImage.rows;

      for(int u=0; u<imgHeight; u++)
        for(int v=0; v<imgWidth; v++){
          int classVal = (int)classImage.at<uchar>(u,v);
          if(sCfg->pSceneObjects[ii]->pObject->objIdx == classVal){
            sCfg->pSceneObjects[ii]->objMask.at<float>(u,v) = 1.0;
            probImage.at<float>(u,v) = 10000;
          }
        }

      cv::imwrite(sCfg->scenePath + "debug_super4PCS/" + sCfg->pSceneObjects[ii]->pObject->objName + ".png", probImage);
    }
  }

  /********************************* function: compute3dSegment ******************************************
  *******************************************************************************************************/

  void Segmentation::compute3dSegment(scene_cfg::SceneCfg *sCfg){

    for(int ii=0; ii<sCfg->numObjects; ii++){
      cv::Mat objDepth = sCfg->depthImage.mul(sCfg->pSceneObjects[ii]->objMask);
      
      utilities::writeDepthImage(objDepth, sCfg->scenePath + "debug_search/" + sCfg->pSceneObjects[ii]->pObject->objName + ".png");

      PointCloudRGB::Ptr segment = PointCloudRGB::Ptr(new PointCloudRGB);
      utilities::convert3dUnOrganizedRGB(objDepth, sCfg->colorImage, sCfg->camIntrinsic, segment);

      clock_t t0 = clock();
      sCfg->pSceneObjects[ii]->pclSegment = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

      copyPointCloud(*segment, *sCfg->pSceneObjects[ii]->pclSegment);

      // pcl::NormalEstimation<pcl::PointXYZRGB, pcl::PointXYZRGBNormal> ne;
      // pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
      // ne.setInputCloud (segment);
      // ne.setSearchMethod (tree);
      // ne.setRadiusSearch (0.005);
      // ne.setViewPoint (0, 0, 0);
      // ne.compute (*sCfg->pSceneObjects[ii]->pclSegment);

      pcl::VoxelGrid<pcl::PointXYZRGB> sor;
      sor.setInputCloud (segment);
      sor.setLeafSize (0.01, 0.01, 0.01);
      sor.filter (*segment);

      pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
      pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointXYZRGBNormal> mls;
      mls.setComputeNormals (true);
      mls.setInputCloud (segment);
      mls.setPolynomialFit (true);
      mls.setSearchMethod (tree);
      mls.setSearchRadius (0.02);
      mls.process (*sCfg->pSceneObjects[ii]->pclSegment);

      std::cout << "number of points: " << segment->points.size() <<std::endl;
      std::cout << "Time after normal estimation: " << float( clock () - t0 ) /  CLOCKS_PER_SEC << std::endl;
    }

  }

  /********************************* end of functions ****************************************************
  *******************************************************************************************************/

}