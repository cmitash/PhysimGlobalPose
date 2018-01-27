#ifndef OBJECTS
#define OBJECTS

#include <common_io.h>
#include <physim_pose_estimation/EstimateObjectPose.h>
#include <physim_pose_estimation/ObjectPose.h>

namespace objects{

	class Objects{
	public:
		Objects(std::string env_p, std::string objName, std::string objType,
				 Eigen::Vector3f symInfo, uchar classId, float modelDiscretization,
				 std::string pcdLocation, std::string objLocation);
		void readPPFMap(std::string env_p, std::string objName);

		int objIdx;
		std::string objName;
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclModel;
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclModelSampled;
		pcl::PolygonMesh objModel;
		Eigen::Vector3f symInfo;
		std::map<std::vector<float>, float> PPFMap;
		float max_count_ppf;
	};
}
#endif