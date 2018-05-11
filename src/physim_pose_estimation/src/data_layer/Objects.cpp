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

		pcl::PointCloud<pcl::PointNormal>::Ptr tmpPclModel_1 = pcl::PointCloud<pcl::PointNormal>::Ptr(new pcl::PointCloud<pcl::PointNormal>);
		pcl::PointCloud<pcl::PointNormal>::Ptr tmpPclModel_2 = pcl::PointCloud<pcl::PointNormal>::Ptr(new pcl::PointCloud<pcl::PointNormal>);
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclModelTmp = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
		
		pclModel = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
		pclModelSampled = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

		pcl::io::loadPLYFile(env_p + "/src/physim_pose_estimation/models_search/" + objName + "/model_search.ply", *tmpPclModel_1);
		pcl::io::loadPLYFile(env_p + "/src/physim_pose_estimation/models_search/" + objName + "/model_validation.ply", *tmpPclModel_2);

		pcl::copyPointCloud(*tmpPclModel_1, *pclModelSampled);
		pcl::copyPointCloud(*tmpPclModel_2, *pclModel);

		pcl::io::loadPolygonFile(env_p + "/src/physim_pose_estimation/models_search/" + objName + "/textured.obj", objModel);
	}

	void Objects::readPPFMap(std::string env_p, std::string objName){
		ifstream ppfFile;
		std::vector<int> ppf_feature(4);
		std::vector<std::pair<int,int> > index_pairs;
		int index1, index2;
		int pair_count;

		ppfFile.open ((env_p + "/src/physim_pose_estimation/models_search/" + objName + "/PPFMap.txt").c_str(), std::ofstream::in);

		while(ppfFile >> ppf_feature[0] >> ppf_feature[1] >> ppf_feature[2] >> ppf_feature[3] >> pair_count){
			index_pairs.clear();
			for(int ii=0; ii<pair_count; ii++){
				ppfFile >> index1 >> index2;
				index_pairs.push_back(std::make_pair(index1, index2));
			}
			PPFMap.insert (std::pair<std::vector<int>, std::vector<std::pair<int,int> > >(ppf_feature, index_pairs));
		}
		std::cout << "PPFMap size is: " << PPFMap.size() << std::endl;
	}
}