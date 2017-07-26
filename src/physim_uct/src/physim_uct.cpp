#include <APCObjects.hpp>
#include <Scene.hpp>

#include <physim_uct/EstimateObjectPose.h>
#include <physim_uct/ObjectPose.h>
#include <geometry_msgs/Pose.h>

// Global definations
std::string env_p;
std::vector<apc_objects::APCObjects*> Objects;

bool estimatePose(physim_uct::EstimateObjectPose::Request &req,
                  physim_uct::EstimateObjectPose::Response &res){
  std::string scenePath(req.SceneFiles);
  scene::Scene *currScene = new scene::Scene(scenePath);
  currScene->performRCNNDetection();

  std::cout<<"number of objects: " << currScene->numObjects << std::endl
           <<"camera pose: " << std::endl << currScene->camPose << std::endl
           <<"camera intrinsics: "<< std::endl << currScene->camIntrinsic << std::endl;

  for(int i=0; i<currScene->numObjects; i++){
    physim_uct::ObjectPose pose;
    geometry_msgs::Pose msg;
    msg.position.x = 0;
    msg.position.y = 0;
    msg.position.z = 0;
    msg.orientation.x = 0;
    msg.orientation.y = 0;
    msg.orientation.z = 0;
    msg.orientation.w = 1;
    pose.label = "ob";
    pose.pose = msg;
    res.Objects.push_back(pose);
  }
  
  return true;
}

// load objects from APC 2016 setting
void loadObjects(std::vector<apc_objects::APCObjects*> &Objects){
  static std::vector<std::string> apc_objects_strs = {
    "crayola_24_ct", "expo_dry_erase_board_eraser", "folgers_classic_roast_coffee",
    "scotch_duct_tape", "up_glucose_bottle", "laugh_out_loud_joke_book", "soft_white_lightbulb",
    "kleenex_tissue_box", "dove_beauty_bar", "elmers_washable_no_run_school_glue", "rawlings_baseball"};

  for(int i=0;i<apc_objects_strs.size();i++){
    apc_objects::APCObjects* tmpobj = new apc_objects::APCObjects(apc_objects_strs[i]);
    Objects.push_back(tmpobj);
  }
}

int main(int argc, char **argv){
  ros::init(argc, argv, "physim_node");
  ros::NodeHandle n;
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