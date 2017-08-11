#include <APCObjects.hpp>
#include <Scene.hpp>
#include <Search.hpp>

#include <physim_uct/EstimateObjectPose.h>
#include <physim_uct/ObjectPose.h>


// mode of operation
bool performSearch = 0;

// Global definations
std::string env_p;
std::vector<apc_objects::APCObjects*> Objects;

// depth_sim package
void initScene (int argc, char **argv);

/********************************* function: estimatePose ***********************************************
********************************************************************************************************/

bool estimatePose(physim_uct::EstimateObjectPose::Request &req,
                  physim_uct::EstimateObjectPose::Response &res){
  std::string scenePath(req.SceneFiles);
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

  if(performSearch){
    search::Search *UCTSearch = new search::Search(currScene);
    UCTSearch->heuristicSearch();

    if(!UCTSearch->bestState->numObjects)
      std::cout<<"Not enough time to search !!!"<<std::endl;
    else{
      std::cout<<"Best State id is: " << UCTSearch->bestState->stateId <<std::endl;
    }

    for(int i=0; i<UCTSearch->bestState->numObjects; i++){
      physim_uct::ObjectPose pose;
      geometry_msgs::Pose msg;

      #ifdef DGB_RESULT
      Eigen::Matrix4f tform;
      PointCloud::Ptr transformedCloud (new PointCloud);
      utilities::convertToMatrix(UCTSearch->bestState->objects[i].second, tform);
      pcl::transformPointCloud(*UCTSearch->bestState->objects[i].first->pclModel, *transformedCloud, tform);
      std::string input1 = scenePath + "debug/result_" + UCTSearch->bestState->objects[i].first->objName + ".ply";
      pcl::io::savePLYFile(input1, *transformedCloud);
      #endif

      Eigen::Vector3d trans = UCTSearch->bestState->objects[i].second.translation();
      Eigen::Quaterniond rot(UCTSearch->bestState->objects[i].second.rotation());

      msg.position.x = trans[0];
      msg.position.y = trans[1];
      msg.position.z = trans[2];
      msg.orientation.x = rot.x();
      msg.orientation.y = rot.y();
      msg.orientation.z = rot.z();
      msg.orientation.w = rot.w();
      pose.label = UCTSearch->bestState->objects[i].first->objName;
      pose.pose = msg;
      res.Objects.push_back(pose);
    }
  } 
  else {
    currScene->getBestSuper4PCS();
    for(int i=0; i<currScene->numObjects; i++){

      #ifdef DGB_RESULT
      Eigen::Matrix4f tform;
      PointCloud::Ptr transformedCloud (new PointCloud);
      utilities::convertToMatrix(currScene->max4PCSPose[i].first, tform);
      std::cout<< tform <<std::endl;
      pcl::transformPointCloud(*currScene->objOrder[i]->pclModel, *transformedCloud, tform);
      std::string input1 = scenePath + "debug/result_" + currScene->objOrder[i]->objName + ".ply";
      pcl::io::savePLYFile(input1, *transformedCloud);
      #endif

      physim_uct::ObjectPose pose;
      geometry_msgs::Pose msg;
      Eigen::Vector3d trans = currScene->max4PCSPose[i].first.translation();
      Eigen::Quaterniond rot(currScene->max4PCSPose[i].first.rotation()); 

      msg.position.x = trans[0];
      msg.position.y = trans[1];
      msg.position.z = trans[2];
      msg.orientation.x = rot.x();
      msg.orientation.y = rot.y();
      msg.orientation.z = rot.z();
      msg.orientation.w = rot.w();
      pose.label = currScene->objOrder[i]->objName;
      pose.pose = msg;
      res.Objects.push_back(pose);
    }
  }
  
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

  ros::ServiceServer service = n.advertiseService("pose_estimation", estimatePose);
  ROS_INFO("Ready for pose estimation");
  ros::spin();

  return 0;
}

/********************************* end of functions ****************************************************
*******************************************************************************************************/