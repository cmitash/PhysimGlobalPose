#ifndef APC_OBJECTS
#define APC_OBJECTS

#include <common_io.h>

namespace apc_objects{

	extern std::map<std::string, int> objMap;
	extern std::string pathPclModels;
	extern std::string pathObjModels;

	class APCObjects{
	public:
		APCObjects(std::string name);

		int objIdx;
		std::string objName;
		PointCloud::Ptr pclModel;
		PointCloud::Ptr pclSegment;
		cv::Rect bbox;
	};
}
extern std::vector<apc_objects::APCObjects*> Objects;
#endif