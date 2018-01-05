#ifndef GLOBAL_CFG
#define GLOBAL_CFG

#include <Objects.hpp>

class GlobalCfg{

public:
	GlobalCfg();
	~GlobalCfg();
	void loadObjects();

	std::string env_p;
	ros::NodeHandle nh;

	int num_objects;
	std::vector<objects::Objects*> gObjects;
};

#endif