#include <ObjectPoseCandidateSet.hpp>

// Super4PCS package
int getProbableTransformsSuper4PCS(std::string input1, std::string input2, std::pair<Eigen::Isometry3d, float> &bestHypothesis, 
            std::vector< std::pair <Eigen::Isometry3d, float> > &hypothesisSet);

namespace pose_candidates{

	ObjectPoseCandidateSet::ObjectPoseCandidateSet(){
		bestHypothesis.first.matrix().setIdentity();
		bestHypothesis.second = 0;
	}

	ObjectPoseCandidateSet::~ObjectPoseCandidateSet(){

	}

	void ObjectPoseCandidateSet::generate(std::string objName, std::string scenePath, 
				PointCloudRGB::Ptr pclSegment, PointCloudRGB::Ptr pclModel){

		if(pclSegment->points.size() <= 30) {
			std::cout << "very few points returned from segmentation !!! returning default pose " << std::endl;
			return;
		}

		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_normals_segment (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_normals_model (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
		pcl::NormalEstimation<pcl::PointXYZRGB, pcl::PointXYZRGBNormal> ne;
		pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());

		std::string input1 = scenePath + "debug_super4PCS/pclSegment_" + objName + ".ply";
		std::string input2 = scenePath + "debug_super4PCS/pclModel_" + objName + ".ply";
		
		copyPointCloud(*pclSegment, *cloud_normals_segment);
		copyPointCloud(*pclModel, *cloud_normals_model);

		ne.setInputCloud (pclSegment);
		ne.setSearchMethod (tree);
		ne.setRadiusSearch (0.01);
		ne.compute (*cloud_normals_segment);

		ne.setInputCloud (pclModel);
		ne.setSearchMethod (tree);
		ne.setRadiusSearch (0.01);
		ne.compute (*cloud_normals_model);

		pcl::io::savePLYFile(input1, *cloud_normals_segment);
		pcl::io::savePLYFile(input2, *cloud_normals_model);

		getProbableTransformsSuper4PCS(input1, input2, bestHypothesis, hypothesisSet);
	}

}
