#include <APCObjects.hpp>
#include <Scene.hpp>
#include <Search.hpp>
#include <UCTSearch.hpp>
#include <Evaluate.hpp>

#include <physim_uct/EstimateObjectPose.h>
#include <physim_uct/ObjectPose.h>

// mode of operation
int generateNewHypothesis = 1;
int performSearch = 0;

// Global definations
std::string env_p;
std::vector<apc_objects::APCObjects*> Objects;
std::map<std::string, Eigen::Vector3f> symMap;

// depth_sim package
void initScene (int argc, char **argv);
void addObjects(pcl::PolygonMesh::Ptr mesh);
void renderDepth(Eigen::Matrix4f pose, cv::Mat &depth_image, std::string path);
void clearScene();

/********************************* function: vizResults *************************************************
********************************************************************************************************/

void vizResults(Eigen::Matrix4f cam_pose, std::string scenePath, apc_objects::APCObjects* obj,
                  Eigen::Isometry3d objPose, cv::Mat &renderedImg, cv::Mat &renderedClassImg, int objId){
  cv::Mat depth_image;
  clearScene();

  pcl::PolygonMesh::Ptr mesh_in (new pcl::PolygonMesh (obj->objModel));
  pcl::PolygonMesh::Ptr mesh_out (new pcl::PolygonMesh (obj->objModel));
  Eigen::Matrix4f transform;
  utilities::convertToMatrix(objPose, transform);
  utilities::convertToWorld(transform, cam_pose);
  utilities::TransformPolyMesh(mesh_in, mesh_out, transform);
  addObjects(mesh_out);
  renderDepth(cam_pose, depth_image, scenePath + "debug_search/renderFinal.png");

  // copy the rendering of the current object over parent state render
  for(int u=0; u<depth_image.rows; u++)
    for(int v=0; v<depth_image.cols; v++){
      float depth_curr = depth_image.at<float>(u,v);
      float depth_parent = renderedImg.at<float>(u,v);
      if(depth_curr > 0 && (depth_parent == 0 || depth_curr < depth_parent)){
        renderedImg.at<float>(u,v) = depth_curr;
        renderedClassImg.at<uchar>(u,v) = objId;
      }
    }
}

/********************************* function: callHeuristicSearch ****************************************
********************************************************************************************************/

static void callHeuristicSearch(scene::Scene *currScene){
  int poseWritten = 0;

  // iterate over all independent search trees
  for(int treeIdx=0; treeIdx<currScene->independentTrees.size(); treeIdx++){
    std::vector< std::vector< std::pair <Eigen::Isometry3d, float> > > hypothesis;

    // iterate over the objects in the tree and create hypothesis set
    for(int jj=0; jj<currScene->independentTrees[treeIdx].size(); jj++){
      hypothesis.push_back(currScene->unconditionedHypothesis[poseWritten + jj]);
    }

    search::Search *HSearch = new search::Search(currScene->independentTrees[treeIdx], currScene->tableParams, hypothesis,
                                  currScene->scenePath, currScene->camPose, currScene->depthImage, currScene->cutOffScore, treeIdx);
    HSearch->heuristicSearch();

    for(int jj=0;jj<currScene->independentTrees[treeIdx].size();jj++){
      currScene->finalState->objects[poseWritten + jj] = HSearch->bestState->objects[jj];
    }

    poseWritten += currScene->independentTrees[treeIdx].size();
  }
}

/********************************* function: callUCTSearch **********************************************
********************************************************************************************************/

static void callUCTSearch(scene::Scene *currScene){
  int poseWritten = 0;

  // iterate over all independent search trees
  for(int treeIdx=0; treeIdx<currScene->independentTrees.size(); treeIdx++){
    std::vector< std::vector< std::pair <Eigen::Isometry3d, float> > > hypothesis;

    // iterate over the objects in the tree and create hypothesis set
    for(int jj=0;jj<currScene->independentTrees[treeIdx].size();jj++){
      hypothesis.push_back(currScene->unconditionedHypothesis[poseWritten + jj]);
    }

    uct_search::UCTSearch *UCTSearch = new uct_search::UCTSearch(currScene->independentTrees[treeIdx], currScene->tableParams, hypothesis,
                                currScene->scenePath, currScene->camPose, currScene->depthImage, currScene->cutOffScore, treeIdx);
    UCTSearch->performSearch();

    for(int jj=0;jj<currScene->independentTrees[treeIdx].size();jj++){
      currScene->finalState->objects[poseWritten + jj] = UCTSearch->bestState->objects[jj];
    }

    poseWritten += currScene->independentTrees[treeIdx].size();

    delete UCTSearch;
  }
}

/********************************* function: estimatePose ***********************************************
********************************************************************************************************/

bool estimatePose(physim_uct::EstimateObjectPose::Request &req,
                  physim_uct::EstimateObjectPose::Response &res){

  std::string scenePath(req.SceneFiles);
  system(("rm -rf " + scenePath + "debug_search").c_str());
  system(("mkdir " + scenePath + "debug_search").c_str());

  if(generateNewHypothesis == 1){
    system(("rm -rf " + scenePath + "debug_super4PCS").c_str());
    system(("mkdir " + scenePath + "debug_super4PCS").c_str());
  }

  scene::Scene *currScene = new scene::Scene(scenePath);
  std::cout<<"number of objects: " << currScene->numObjects << std::endl
           <<"camera pose: " << std::endl << currScene->camPose << std::endl
           <<"camera intrinsics: "<< std::endl << currScene->camIntrinsic << std::endl;

  // Remove the table
  currScene->removeTable();
  currScene->getTableParams();

  // Perform RCNN detection and extract corresponding segment
  // currScene->performRCNNDetection();
  // currScene->get3DSegments();

  // Use ground truth segmentation result
  currScene->getGTSegments();
  currScene->getSegmentationPrior();

  // Use ground truth object order
  currScene->getOrder();

  // Whether to generate a new set of pose hypotheses
  if(generateNewHypothesis == 1)
    currScene->computeHypothesisSet();
  else
    currScene->readHypothesis();

  // call search routine
  if(performSearch == 1)
    callHeuristicSearch(currScene);
  else if(performSearch == 2)
    callUCTSearch(currScene);

  // copy final results to rosmessage
  cv::Mat renderedImg = cv::Mat::zeros(480, 640, CV_32FC1);
  cv::Mat renderedClassImg = cv::Mat::zeros(480, 640, CV_8UC1);

  system(("rm " + scenePath + "result.txt").c_str());

  // iterate over scene objects
  for(int ii=0; ii<currScene->finalState->numObjects; ii++){
    physim_uct::ObjectPose pose;
    geometry_msgs::Pose msg;

    #ifdef DGB_RESULT
    Eigen::Matrix4f tform;
    PointCloud::Ptr transformedCloud (new PointCloud);
    utilities::convertToMatrix(currScene->finalState->objects[ii].second, tform);
    pcl::transformPointCloud(*currScene->finalState->objects[ii].first->pclModel, *transformedCloud, tform);
    std::string input1 = scenePath + "debug_search/result_" + currScene->finalState->objects[ii].first->objName + ".ply";

    PointCloudRGB::Ptr cloud_xyzrgb (new PointCloudRGB);
    cloud_xyzrgb->points.resize(transformedCloud->size());
    for (size_t i = 0; i < transformedCloud->points.size(); i++) {
        cloud_xyzrgb->points[i].x = transformedCloud->points[i].x;
        cloud_xyzrgb->points[i].y = transformedCloud->points[i].y;
        cloud_xyzrgb->points[i].z = transformedCloud->points[i].z;
        cloud_xyzrgb->points[i].r = 0;
        cloud_xyzrgb->points[i].g = 0;
        cloud_xyzrgb->points[i].b = 1;
    }
    pcl::io::savePLYFile(input1, *cloud_xyzrgb);
    #endif

    // To visualize the results
    vizResults(currScene->camPose, scenePath, currScene->finalState->objects[ii].first,
      currScene->finalState->objects[ii].second, renderedImg, renderedClassImg, ii+1);
    utilities::writeDepthImage(renderedImg, scenePath + "debug_search/renderFinalDepth.png");
    utilities::writeClassImage(renderedClassImg, currScene->colorImage, scenePath + "debug_search/renderFinalClass.png");

    Eigen::Matrix4f finalPoseMat;
    Eigen::Isometry3d finalPoseIsometric;
    utilities::convertToMatrix(currScene->finalState->objects[ii].second, finalPoseMat);
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
    pose.label = currScene->finalState->objects[ii].first->objName;
    pose.pose = msg;
    res.Objects.push_back(pose);

    ofstream pFile;
    pFile.open ((scenePath + "result.txt").c_str(), std::ofstream::out | std::ofstream::app);
    pFile << trans[0] << ", " << trans[1] << ", " << trans[2] 
      << ", " << rot.w() << ", " << rot.x() << ", " << rot.y() << ", " << rot.z() << std::endl;
    pFile.close();
  }
  
  delete currScene;

  return true;
}

/********************************* function: loadObjects ***********************************************
********************************************************************************************************/

void loadObjects(std::vector<apc_objects::APCObjects*> &Objects){
  static std::vector<std::string> apc_objects_strs;
  apc_objects_strs.push_back("crayola_24_ct");
  apc_objects_strs.push_back("expo_dry_erase_board_eraser");
  apc_objects_strs.push_back("folgers_classic_roast_coffee");
  apc_objects_strs.push_back("scotch_duct_tape");
  apc_objects_strs.push_back("up_glucose_bottle");
  apc_objects_strs.push_back("laugh_out_loud_joke_book");
  apc_objects_strs.push_back("soft_white_lightbulb");
  apc_objects_strs.push_back("kleenex_tissue_box");
  apc_objects_strs.push_back("dove_beauty_bar");
  apc_objects_strs.push_back("elmers_washable_no_run_school_glue");
  apc_objects_strs.push_back("rawlings_baseball");

  symMap["crayola_24_ct"] << 180,180,180;
  symMap["expo_dry_erase_board_eraser"] << 180,180,180;
  symMap["folgers_classic_roast_coffee"] << 360,180,180;
  symMap["scotch_duct_tape"] << 180,180,360;
  symMap["dasani_water_bottle"] << 360,0,0;
  symMap["jane_eyre_dvd"] << 180,180,180;
  symMap["up_glucose_bottle"] << 360,0,0;
  symMap["laugh_out_loud_joke_book"] << 180,180,180;
  symMap["soft_white_lightbulb"] << 90,180,180;
  symMap["kleenex_tissue_box"] << 90,90,90;
  symMap["ticonderoga_12_pencils"] << 180,180,180;
  symMap["dove_beauty_bar"] << 180,180,180;
  symMap["dr_browns_bottle_brush"] << 180,0,0;
  symMap["elmers_washable_no_run_school_glue"] << 180,0,0;
  symMap["rawlings_baseball"] << 360,360,360;

  for(int i=0;i<apc_objects_strs.size();i++){
    apc_objects::APCObjects* tmpobj = new apc_objects::APCObjects(apc_objects_strs[i]);
    Objects.push_back(tmpobj);
  }
}

/********************************* function: main *******************************************************
********************************************************************************************************/

int main(int argc, char **argv){
  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

  ros::init(argc, argv, "physim_node");
  ros::NodeHandle n;
  
  // Initialize openGL for rendering
  initScene (0, NULL);

  if(const char* env = std::getenv("PHYSIM_GLOBAL_POSE")){
      std::cout << "Your PHYSIM_GLOBAL_POSE repository is: " << env << std::endl;
      env_p = std::string(env);
  }
  else{
    std::cout<<"Please set PHYSIM_GLOBAL_POSE in bashrc"<<std::endl;
    exit(-1);
  }
  loadObjects(Objects);

  ros::ServiceServer service = n.advertiseService("pose_estimation", estimatePose);
  ROS_INFO("Ready for pose estimation");
  ros::spin();

  return 0;
}

/********************************* end of functions ****************************************************
*******************************************************************************************************/