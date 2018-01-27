#include <ObjectPoseCandidateSet.hpp>

// Super4PCS package
int getProbableTransformsSuper4PCS(std::string input1, std::string input2, std::string input3, std::pair<Eigen::Isometry3d, float> &bestHypothesis, 
            std::vector< std::pair <Eigen::Isometry3d, float> > &hypothesisSet, std::string probImagePath, 
            std::map<std::vector<float>, float> PPFMap, float max_count_ppf, Eigen::Matrix3f camIntrinsic, std::string objName);

namespace pose_candidates{

	ObjectPoseCandidateSet::ObjectPoseCandidateSet(){
		bestHypothesis.first.matrix().setIdentity();
		bestHypothesis.second = 0;
	}

	ObjectPoseCandidateSet::~ObjectPoseCandidateSet(){

	}

	void ObjectPoseCandidateSet::generate(std::string objName, std::string scenePath, 
				pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclSegment, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclModel, 
				pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclModelSampled, 
				std::map<std::vector<float>, float> PPFMap, float max_count_ppf, Eigen::Matrix3f camIntrinsic){

		if(pclSegment->points.size() <= 30) {
			std::cout << "very few points returned from segmentation !!! returning default pose " << std::endl;
			return;
		}

		pcl::VoxelGrid<pcl::PointXYZRGBNormal> sor;
		sor.setInputCloud (pclSegment);
		sor.setLeafSize (0.01, 0.01, 0.01);
		sor.filter (*pclSegment);

		std::string input1 = scenePath + "debug_super4PCS/pclSegment_" + objName + ".ply";
		std::cout << input1 << std::endl;
		pcl::io::savePLYFile(input1, *pclSegment);

		std::string input2 = scenePath + "debug_super4PCS/pclModel_" + objName + ".ply";
		std::cout << input2 << std::endl;
		pcl::io::savePLYFile(input2, *pclModel);

		std::string input3 = scenePath + "debug_super4PCS/pclModelSampled_" + objName + ".ply";
		std::cout << input3 << std::endl;
		pcl::io::savePLYFile(input3, *pclModelSampled);

		std::string probImagePath = scenePath + "debug_super4PCS/" + objName + ".png";

		getProbableTransformsSuper4PCS(input1, input2, input3, bestHypothesis, hypothesisSet, probImagePath, PPFMap, max_count_ppf, camIntrinsic, objName);
		std::cout << bestHypothesis.first.matrix() << std::endl;
		
	}

}
