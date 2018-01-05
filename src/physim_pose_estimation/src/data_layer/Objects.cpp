#include <Objects.hpp>

namespace objects{

	/********************************* function: constructor ***********************************************
	*******************************************************************************************************/

	Objects::Objects(std::string env_p, std::string objName, std::string objType,
				 Eigen::Vector3f symInfo, uchar classId, float modelDiscretization,
				 std::string pcdLocation, std::string objLocation) {
		this->objName = objName;
		this->objIdx = classId;
		this->symInfo = symInfo;

		PointCloud::Ptr tmpPclModel = PointCloud::Ptr(new PointCloud);
		pclModel = PointCloudRGB::Ptr(new PointCloudRGB);

		pcl::io::loadPCDFile(env_p + "/models/" + pcdLocation, *tmpPclModel);
		pcl::copyPointCloud(*tmpPclModel, *pclModel);
		
		if(classId == 1 || classId==6 || classId==14 || classId==19 || classId==20)
			pcl::io::loadPolygonFile(env_p + "/models/" + objLocation , objModel);

		pcl::VoxelGrid<pcl::PointXYZRGB> sor;
		sor.setInputCloud (pclModel);
		sor.setLeafSize (modelDiscretization, modelDiscretization, modelDiscretization);
		sor.filter (*pclModel);
	}	
}