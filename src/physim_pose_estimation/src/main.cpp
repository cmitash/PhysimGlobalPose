#include <GlobalCfg.hpp>
#include <SceneCfg.hpp>

// global config pointer
GlobalCfg *pCfg;

// depth_sim package
void initScene (int argc, char **argv);
void addObjects(pcl::PolygonMesh::Ptr mesh);
void renderDepth(Eigen::Matrix4f pose, cv::Mat &depth_image, std::string path);
void clearScene();

/********************************* function: estimatePose ***********************************************
********************************************************************************************************/

bool estimatePose(physim_pose_estimation::EstimateObjectPose::Request &req,
                  physim_pose_estimation::EstimateObjectPose::Response &res){

  // Initialize the scene based on the type of dataset or camera input is chosen as default
  scene_cfg::SceneCfg *currScene;
  if(!req.OperationMode.compare("APC"))
    currScene = new scene_cfg::APCSceneCfg(req.SceneFiles, req.SegmentationMode, req.HypothesisGenerationMode, req.HypothesisVerificationMode);
  else if(!req.OperationMode.compare("YCB"))
    currScene = new scene_cfg::YCBSceneCfg(req.SceneFiles, req.SegmentationMode, req.HypothesisGenerationMode, req.HypothesisVerificationMode);
  else
    currScene = new scene_cfg::CAMSceneCfg(req.SceneFiles, req.SegmentationMode, req.HypothesisGenerationMode, req.HypothesisVerificationMode);

  currScene->cleanDebugLocations();
  currScene->getSceneInfo(pCfg);

  std::cout<<"number of objects: " << currScene->numObjects << std::endl
           <<"camera pose: " << std::endl << currScene->camPose << std::endl
           <<"camera intrinsics: "<< std::endl << currScene->camIntrinsic << std::endl;

  currScene->removeTable();
  currScene->getTableParams();
  currScene->perfromSegmentation(pCfg);
  currScene->generateHypothesis();
  currScene->performHypothesisSelection();

  // iterate over scene objects
  for(int ii=0; ii<currScene->numObjects; ii++){

    physim_pose_estimation::ObjectPose pose;
    geometry_msgs::Pose msg;
    Eigen::Matrix4f finalPoseMat;
    Eigen::Isometry3d finalPoseIsometric;

    // Convert the pose to global frame
    utilities::convertToMatrix(currScene->pSceneObjects[ii]->objPose, finalPoseMat);
    utilities::convertToWorld(finalPoseMat, currScene->camPose);
    utilities::convertToIsometry3d(finalPoseMat, finalPoseIsometric);
    Eigen::Vector3d trans = finalPoseIsometric.translation();
    Eigen::Quaterniond rot(finalPoseIsometric.rotation());

    msg.position.x = trans[0];
    msg.position.y = trans[1];
    msg.position.z = trans[2];
    msg.orientation.x = rot.x();
    msg.orientation.y = rot.y();
    msg.orientation.z = rot.z();
    msg.orientation.w = rot.w();
    pose.label = currScene->pSceneObjects[ii]->pObject->objName;
    pose.pose = msg;
    res.Objects.push_back(pose);

    // Write final result in the file result.txt
    ofstream pFile;
    pFile.open ((currScene->scenePath + "result.txt").c_str(), std::ofstream::out | std::ofstream::app);
    pFile << pose.label << " " << msg.position.x << " " << msg.position.y << " " << msg.position.z 
      << " " << msg.orientation.w << " " << msg.orientation.x << " " << msg.orientation.y << " " << msg.orientation.z << std::endl;
    pFile.close();
  }
  
  delete currScene;

  return true;
}

/********************************* function: main *******************************************************
********************************************************************************************************/

int main(int argc, char **argv){
  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
  ros::init(argc, argv, "physim_node");
  pCfg = new GlobalCfg();
  initScene (0, NULL); // Initialize openGL for rendering

  pCfg->loadObjects();

  ros::ServiceServer service = pCfg->nh.advertiseService("pose_estimation", estimatePose);
  ROS_INFO("Ready for pose estimation");
  ros::spin();

  return 0;
}

/********************************* end of functions ****************************************************
*******************************************************************************************************/