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

  if(performSearch){
    search::Search *UCTSearch = new search::Search(currScene);
    UCTSearch->heuristicSearch();

    if(!UCTSearch->bestState->numObjects)
      std::cout<<"Not enough time to search !!!"<<std::endl;
    else{
      std::cout<<"Best State id is: " << UCTSearch->bestState->stateId <<std::endl;

      #ifdef DBG_SUPER4PCS
      ofstream scoreFile;
      scoreFile.open ((currScene->scenePath + "debug/scores.txt").c_str(), std::ofstream::out | std::ofstream::app);
      scoreFile << UCTSearch->bestState->stateId << " " << UCTSearch->bestState->score << std::endl;
      scoreFile.close();

      for(int i = 0;i < UCTSearch->bestState->objects.size();i++){
        Eigen::Matrix4f tform;
        utilities::convertToMatrix(UCTSearch->bestState->objects[i].second, tform);
        utilities::convertToWorld(tform, currScene->camPose);
        ifstream gtPoseFile;
        Eigen::Matrix4f gtPose;
        gtPose.setIdentity();
        gtPoseFile.open((currScene->scenePath + "gt_pose_" + UCTSearch->bestState->objects[i].first->objName + ".txt").c_str(), std::ifstream::in);
        gtPoseFile >> gtPose(0,0) >> gtPose(0,1) >> gtPose(0,2) >> gtPose(0,3) 
             >> gtPose(1,0) >> gtPose(1,1) >> gtPose(1,2) >> gtPose(1,3)
             >> gtPose(2,0) >> gtPose(2,1) >> gtPose(2,2) >> gtPose(2,3);
        gtPoseFile.close();
        float rotErr, transErr;
        utilities::getPoseError(tform, gtPose, UCTSearch->bestState->objects[i].first->symInfo, rotErr, transErr);
        ofstream statsFile;
        statsFile.open ((currScene->scenePath + "debug/stats_" + UCTSearch->bestState->objects[i].first->objName + ".txt").c_str(), std::ofstream::out | std::ofstream::app);
        statsFile << "afterSearchRotErr: " << rotErr << std::endl;
        statsFile << "afterSearchTransErr: " << transErr << std::endl;
        statsFile.close();
      }
      #endif
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
    for(int i=0; i<currScene->numObjects; i++){

      #ifdef DGB_RESULT
      Eigen::Matrix4f tform;
      PointCloud::Ptr transformedCloud (new PointCloud);
      utilities::convertToMatrix(currScene->max4PCSPose[i].first, tform);
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

  ros::ServiceServer service = n.advertiseService("pose_estimation", estimatePose);
  ROS_INFO("Ready for pose estimation");
  ros::spin();

  return 0;
}

/********************************* end of functions ****************************************************
*******************************************************************************************************/