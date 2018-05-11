#include <GlobalCfg.hpp>
#include <SceneCfg.hpp>

// global config pointer
GlobalCfg *pCfg;

namespace utilities{
  std::map<std::string, geometry_msgs::Pose> anyTimePoseArray;
  PointCloudRGB::Ptr pc_viz;
}

int runVizThread = 1;

// depth_sim package
void initScene (int argc, char **argv);
void addObjects(pcl::PolygonMesh::Ptr mesh);
void renderDepth(Eigen::Matrix4f pose, cv::Mat &depth_image, std::string path);
void clearScene();

void publishMarkers(std::vector<visualization_msgs::Marker> &marker, std::vector<ros::Publisher> &marker_pub, ros::Publisher pub) {
  while(runVizThread){
    for (int ii=0; ii<pCfg->num_objects; ii++){
      std::map<std::string, geometry_msgs::Pose>::iterator it = utilities::anyTimePoseArray.find(pCfg->gObjects[ii]->objName);
      geometry_msgs::Pose msg = it->second;
      marker[ii].pose.position.x = msg.position.x;
      marker[ii].pose.position.y = msg.position.y;
      marker[ii].pose.position.z = msg.position.z;
      marker[ii].pose.orientation.x = msg.orientation.x;
      marker[ii].pose.orientation.y = msg.orientation.y;
      marker[ii].pose.orientation.z = msg.orientation.z;
      marker[ii].pose.orientation.w = msg.orientation.w;
      marker_pub[ii].publish(marker[ii]);
    }

    utilities::pc_viz->header.frame_id = "/world";
    pub.publish (utilities::pc_viz);
    ros::Duration(0.1).sleep();
  }
}

void initMarkers(visualization_msgs::Marker &marker, ros::Publisher &marker_pub, std::string objName, int objId) {
  char markerId[30];
  sprintf(markerId, "Object%04d", objId);

  // marker_pub = pCfg->nh.advertise<visualization_msgs::Marker>(objName.c_str(), 1);
  marker_pub = pCfg->nh.advertise<visualization_msgs::Marker>(markerId, 1);

  // Set the frame ID and timestamp.  See the TF tutorials for information on these.
  marker.header.frame_id = "/world";
  marker.header.stamp = ros::Time::now();

  // Set the namespace and id for this marker.  This serves to create a unique ID
  // Any marker sent with the same namespace and id will overwrite the old one
  marker.ns = "basic_shapes";
  marker.id = 0;

  // Set the marker type
  marker.type = visualization_msgs::Marker::MESH_RESOURCE;

  // Set the marker action.  Options are ADD and DELETE
  marker.action = visualization_msgs::Marker::ADD;
  
  // Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
  marker.pose.position.x = 10;
  marker.pose.position.y = 10;
  marker.pose.position.z = 10;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;

  // Set the scale of the marker -- 1x1x1 here means 1m on a side
  marker.scale.x = 1.0;
  marker.scale.y = 1.0;
  marker.scale.z = 1.0;

  marker.mesh_use_embedded_materials = true;

  marker.lifetime = ros::Duration();
  marker.mesh_resource = "package://physim_pose_estimation/models_visualization/" + objName + ".ply";
}

/********************************* function: estimatePose ***********************************************
********************************************************************************************************/

bool estimatePose(physim_pose_estimation::EstimateObjectPose::Request &req,
                  physim_pose_estimation::EstimateObjectPose::Response &res){

  // refresh visualization
  for(int ii=0; ii<pCfg->num_objects; ii++) {
    std::map<std::string, geometry_msgs::Pose>::iterator it = utilities::anyTimePoseArray.find(pCfg->gObjects[ii]->objName);
    it->second.position.x = 10;
    it->second.position.y = 10;
    it->second.position.z = 10;
    it->second.orientation.x = 0;
    it->second.orientation.y = 0;
    it->second.orientation.z = 0;
    it->second.orientation.w = 1;
  }

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
  currScene->perfromSegmentation(pCfg);
  
  clock_t time_start = clock ();
  
  currScene->generateHypothesis();
  currScene->performHypothesisSelection();

  float total_time = float( clock () - time_start ) /  CLOCKS_PER_SEC;

  // ofstream pFile;
  // pFile.open ("/media/chaitanya/DATADRIVE0/datasets/YCB_Video_Dataset/time.txt", std::ofstream::out | std::ofstream::app);
  // pFile << total_time << std::endl;
  // pFile.close();
  
  copyPointCloud(*currScene->sceneCloud, *utilities::pc_viz);

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

  // initializing markers
  std::vector<ros::Publisher> marker_pubs(pCfg->num_objects); 
  std::vector<visualization_msgs::Marker> markers(pCfg->num_objects);
  for(int ii=0; ii<pCfg->num_objects; ii++) {
    initMarkers(markers[ii], marker_pubs[ii], pCfg->gObjects[ii]->objName, ii);
    geometry_msgs::Pose identity_pose;
    identity_pose.position.x = 10;
    identity_pose.position.y = 10;
    identity_pose.position.z = 10;
    identity_pose.orientation.x = 0;
    identity_pose.orientation.y = 0;
    identity_pose.orientation.z = 0;
    identity_pose.orientation.w = 1;
    utilities::anyTimePoseArray.insert(std::make_pair(pCfg->gObjects[ii]->objName, identity_pose));
  }
  ros::Publisher pub = pCfg->nh.advertise<PointCloudRGB> ("scene_cloud", 1);
  PointCloudRGB::Ptr DummySceneCloud(new PointCloudRGB);
  DummySceneCloud->points.push_back (pcl::PointXYZRGB(1.0, 2.0, 3.0));
  utilities::pc_viz = PointCloudRGB::Ptr(new PointCloudRGB);
  utilities::pc_viz->resize(1);
  copyPointCloud(*DummySceneCloud, *utilities::pc_viz);
  utilities::pc_viz->header.frame_id = "/world";

  std::thread marker_thread (publishMarkers, std::ref(markers), std::ref(marker_pubs), pub);

  ros::ServiceServer service = pCfg->nh.advertiseService("pose_estimation", estimatePose);
  ROS_INFO("Ready for pose estimation");
  ros::spin();
  runVizThread = 0;

  marker_thread.join();
  return 0;
}

/********************************* end of functions ****************************************************
*******************************************************************************************************/