#include <APCObjects.hpp>
#include <Scene.hpp>
#include <Search.hpp>
#include <Evaluate.hpp>

#include <physim_uct/EstimateObjectPose.h>
#include <physim_uct/ObjectPose.h>

// mode of operation
bool performSearch = 1;
bool evalPoseDataset17 = 0;
int evalDatasetSize = 30; 

// Global definations
std::string env_p;
std::vector<apc_objects::APCObjects*> Objects;
std::map<std::string, Eigen::Vector3f> symMap;

// depth_sim package
void initScene (int argc, char **argv);

/********************************* function: estimatePose ***********************************************
********************************************************************************************************/

bool estimatePose(physim_uct::EstimateObjectPose::Request &req,
                  physim_uct::EstimateObjectPose::Response &res){
  std::string scenePath(req.SceneFiles);
  system(("rm -rf " + scenePath + "debug").c_str());
  system(("mkdir " + scenePath + "debug").c_str());

  scene::Scene *currScene = new scene::Scene(scenePath);
  std::cout<<"number of objects: " << currScene->numObjects << std::endl
           <<"camera pose: " << std::endl << currScene->camPose << std::endl
           <<"camera intrinsics: "<< std::endl << currScene->camIntrinsic << std::endl;

  // Initialize the scene
  currScene->removeTable();
  currScene->performRCNNDetection();
  currScene->get3DSegments();
  currScene->getOrder();
  currScene->getUnconditionedHypothesis();

  // perform search if the flag is set
  if(performSearch){
    int poseWritten = 0;
    for(int ii=0;ii<currScene->independentTrees.size();ii++){
      
      std::vector< std::vector< std::pair <Eigen::Isometry3d, float> > > hypothesis;
      for(int jj=0;jj<currScene->independentTrees[ii].size();jj++)
        hypothesis.push_back(currScene->unconditionedHypothesis[poseWritten + jj]);

      search::Search *UCTSearch = new search::Search(currScene->independentTrees[ii], hypothesis,
                                  currScene->scenePath, currScene->camPose, currScene->depthImage, currScene->cutOffScore);
      UCTSearch->heuristicSearch();

      for(int jj=0;jj<currScene->independentTrees[ii].size();jj++)
        currScene->finalState->objects[poseWritten + jj] = UCTSearch->bestState->objects[jj];

      poseWritten += currScene->independentTrees[ii].size();
    }

    cv::Mat depth_image_final;
    currScene->finalState->updateStateId(-2);
    currScene->finalState->render(currScene->camPose, currScene->scenePath, depth_image_final);
  }

  // copy final results to rosmessage
  for(int ii=0; ii<currScene->finalState->numObjects; ii++){
    physim_uct::ObjectPose pose;
    geometry_msgs::Pose msg;

    #ifdef DGB_RESULT
    Eigen::Matrix4f tform;
    PointCloud::Ptr transformedCloud (new PointCloud);
    utilities::convertToMatrix(currScene->finalState->objects[ii].second, tform);
    pcl::transformPointCloud(*currScene->finalState->objects[ii].first->pclModel, *transformedCloud, tform);
    std::string input1 = scenePath + "debug/result_" + currScene->finalState->objects[ii].first->objName + ".ply";
    pcl::io::savePLYFile(input1, *transformedCloud);
    #endif

    Eigen::Vector3d trans = currScene->finalState->objects[ii].second.translation();
    Eigen::Quaterniond rot(currScene->finalState->objects[ii].second.rotation());

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
  }
  
  return true;
}

/********************************* function: evaluatePoseDataset17 **************************************
********************************************************************************************************/

void evaluatePoseDataset17(){
  evaluate::Evaluate *pEval = new evaluate::Evaluate();

  for(int i=1; i<=evalDatasetSize; i++){
    char buf[50];
    sprintf(buf,"/home/chaitanya/PoseDataset17/table/scene-%04d/", i);
    std::string scenePath(buf);
    std::cout << scenePath << std::endl;
    scene::Scene *currScene = new scene::Scene(scenePath);
    std::cout<<"number of objects: " << currScene->numObjects << std::endl
             <<"camera pose: " << std::endl << currScene->camPose << std::endl
             <<"camera intrinsics: "<< std::endl << currScene->camIntrinsic << std::endl;
    currScene->removeTable();
    currScene->performRCNNDetection();
    currScene->get3DSegments();
    currScene->getOrder();
    pEval->readGroundTruth(currScene);
    pEval->getSuper4pcsError(currScene, i-1);
    pEval->getAllHypoError(currScene, i-1);
    pEval->getClusterHypoError(currScene, i-1);
    pEval->getSearchResults(currScene, i-1);
    
    delete currScene;
  }
  pEval->writeResults();
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

  if(evalPoseDataset17){
    evaluatePoseDataset17();
    exit(-1);
  }

  ros::ServiceServer service = n.advertiseService("pose_estimation", estimatePose);
  ROS_INFO("Ready for pose estimation");
  ros::spin();

  return 0;
}

/********************************* end of functions ****************************************************
*******************************************************************************************************/