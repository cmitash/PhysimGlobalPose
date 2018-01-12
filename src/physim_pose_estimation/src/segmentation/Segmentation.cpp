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
        cv::Mat sumImg = cv::Mat::zeros(480, 640, CV_16UC1);
        for(int ii=0; ii<sCfg->numObjects; ii++){
          cv::Mat probImage = cv::imread(sCfg->scenePath + "debug_super4PCS/" + sCfg->pSceneObjects[ii]->pObject->objName + ".png", CV_16UC1);
          cv::add(probImage, sumImg, sumImg);
        }

        for(int ii=0; ii<sCfg->numObjects; ii++){
          cv::Mat probImage = cv::imread(sCfg->scenePath + "debug_super4PCS/" + sCfg->pSceneObjects[ii]->pObject->objName + ".png", CV_16UC1);
          int imgWidth = probImage.cols;
          int imgHeight = probImage.rows;

          for(int u=0; u<imgHeight; u++){
            for(int v=0; v<imgWidth; v++) {
              if(!(float)sumImg.at<unsigned short>(u,v))continue;
              float probVal = (float)probImage.at<unsigned short>(u,v);
              probVal = probVal/(float)sumImg.at<unsigned short>(u,v);
              if(probVal > 0.4)
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

      int imgWidth = classImage.cols;
      int imgHeight = classImage.rows;

      for(int u=0; u<imgHeight; u++)
        for(int v=0; v<imgWidth; v++){
          int classVal = (int)classImage.at<uchar>(u,v);
          if(sCfg->pSceneObjects[ii]->pObject->objIdx == classVal){
            sCfg->pSceneObjects[ii]->objMask.at<float>(u,v) = 1.0;
          }
        }
    }
  }

  /********************************* function: compute3dSegment ******************************************
  *******************************************************************************************************/

  void Segmentation::compute3dSegment(scene_cfg::SceneCfg *sCfg){

    for(int ii=0; ii<sCfg->numObjects; ii++){
      cv::Mat objDepth = sCfg->depthImage.mul(sCfg->pSceneObjects[ii]->objMask);
      
      utilities::writeDepthImage(objDepth, sCfg->scenePath + "debug_search/" + sCfg->pSceneObjects[ii]->pObject->objName + ".png");

      sCfg->pSceneObjects[ii]->pclSegment = PointCloudRGB::Ptr(new PointCloudRGB);
      utilities::convert3dUnOrganizedRGB(objDepth, sCfg->colorImage, sCfg->camIntrinsic, sCfg->pSceneObjects[ii]->pclSegment);

      pcl::VoxelGrid<pcl::PointXYZRGB> sor;
      sor.setInputCloud (sCfg->pSceneObjects[ii]->pclSegment);
      sor.setLeafSize (0.005f, 0.005f, 0.005f);
      sor.filter (*sCfg->pSceneObjects[ii]->pclSegment);
    }

  }

  /********************************* end of functions ****************************************************
  *******************************************************************************************************/

}