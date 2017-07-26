#ifndef APC_OBJECTS
#define APC_OBJECTS

#include <common_io.h>

namespace apc_objects{

	extern std::map<std::string, int> objMap;
	extern std::string path_pcl_models;
	extern std::string path_obj_models;

	class APCObjects{
	public:
		APCObjects(std::string name);
		std::string objName;
		int objIdx;
		PointCloud::Ptr pcl_model;
		cv::Rect bbox;
	};
}
extern std::vector<apc_objects::APCObjects*> Objects;
#endif