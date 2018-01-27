#include <GlobalCfg.hpp>

/********************************* function: constructor ***********************************************
*******************************************************************************************************/

GlobalCfg::GlobalCfg(){

	// get repository path
	if(const char* env = std::getenv("PHYSIM_GLOBAL_POSE")) {
      std::cout << "Your PHYSIM_GLOBAL_POSE repository is: " << env << std::endl;
      env_p = std::string(env);
  	} else {
		std::cout<<"Please set PHYSIM_GLOBAL_POSE in bashrc"<<std::endl;
	    exit(-1);
	}
}

/********************************* function: destructor ***********************************************
*******************************************************************************************************/
GlobalCfg::~GlobalCfg(){
	for(int ii=0; ii<num_objects; ii++){
		delete gObjects[ii];
	}
}

/********************************* function: loadObjects ***********************************************
Read objects from the config file and load them
********************************************************************************************************/

void GlobalCfg::loadObjects(){
	float modelDiscretization;

	system(("rosparam load " + env_p + "/src/physim_pose_estimation/src/data_layer/obj_config.yml").c_str());
	nh.getParam("/objects/num_objects", num_objects);
	nh.getParam("/objects/modelDiscretization", modelDiscretization);

	for(int ii=0; ii<num_objects; ii++){
		char objTopic[50];
		std::string obj_name;
		std::string obj_type;
		int classId;
		std::vector<float> symmetry(3);
		std::string pcdLocation;
		std::string objLocation;
		
		sprintf(objTopic, "/objects/object_%d", ii+1);
		nh.getParam((std::string(objTopic) + "/name").c_str(), obj_name);
		nh.getParam((std::string(objTopic) + "/type").c_str(), obj_type);
		nh.getParam((std::string(objTopic) + "/classId").c_str(), classId);
		nh.getParam((std::string(objTopic) + "/symmetry").c_str(), symmetry);
		nh.getParam((std::string(objTopic) + "/location_obj").c_str(), objLocation);
		nh.getParam((std::string(objTopic) + "/location_pcd").c_str(), pcdLocation);

		std::cout << "Loaded Object " << ii+1 << " : " << obj_name << ", " << obj_type << ", " << classId << std::endl;
		Eigen::Vector3f symInfo(symmetry[0], symmetry[1], symmetry[2]);
		
		objects::Objects *tmpObj = new objects::Objects(env_p, obj_name, obj_type, symInfo, classId, 
											modelDiscretization, pcdLocation, objLocation);

		tmpObj->readPPFMap(env_p, obj_name);

		gObjects.push_back(tmpObj);
	}
}
