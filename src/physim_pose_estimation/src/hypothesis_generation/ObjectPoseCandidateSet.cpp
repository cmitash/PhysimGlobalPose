#include <ObjectPoseCandidateSet.hpp>

// Super4PCS package
void getProbableTransformsSuper4PCS(std::string input1, std::string input2, std::string input3, std::pair<Eigen::Isometry3d, float> &bestHypothesis, 
            std::vector< std::pair <Eigen::Isometry3d, float> > &hypothesisSet, std::string probImagePath, 
            std::map<std::vector<int>, std::vector<std::pair<int,int> > > &PPFMap, int max_count_ppf, Eigen::Matrix3f camIntrinsic, std::string objName, std::string scenePath);

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
				std::map<std::vector<int>, std::vector<std::pair<int,int> > > &PPFMap, int max_count_ppf, Eigen::Matrix4f camPose, Eigen::Matrix3f camIntrinsic/*, std::thread &th_id*/){

		// pcl::VoxelGrid<pcl::PointXYZRGBNormal> sor;
		// sor.setInputCloud (pclSegment);
		// sor.setLeafSize (0.01, 0.01, 0.01);
		// sor.filter (*pclSegment);

		pcl::RadiusOutlierRemoval<pcl::PointXYZRGBNormal> outrem;
	    outrem.setInputCloud(pclSegment);
	    outrem.setRadiusSearch(0.03);
	    outrem.setMinNeighborsInRadius (10);
	    outrem.filter (*pclSegment);

	    if(pclSegment->points.size() <= 30) {
			std::cout << "very few points returned from segmentation !!! returning default pose " << std::endl;
			return;
		}

	    for (int ii=0; ii< pclSegment->points.size(); ii++) {
	    	pcl::flipNormalTowardsViewpoint (pclSegment->points[ii], 
	    		0, 0, 0, 
		        pclSegment->points[ii].normal[0], 
		        pclSegment->points[ii].normal[1], 
		        pclSegment->points[ii].normal[2]);
	    	float magnitude = std::sqrt(pclSegment->points[ii].normal[0]*pclSegment->points[ii].normal[0] + 
	    						pclSegment->points[ii].normal[1]*pclSegment->points[ii].normal[1] + 
	    						pclSegment->points[ii].normal[2]*pclSegment->points[ii].normal[2]);
	    	pclSegment->points[ii].normal[0] /= magnitude;
	    	pclSegment->points[ii].normal[1] /= magnitude;
	    	pclSegment->points[ii].normal[2] /= magnitude;
	    }
	    

		std::string input1 = scenePath + "debug_super4PCS/pclSegment_" + objName + ".ply";
		pcl::io::savePLYFile(input1, *pclSegment);

		std::string input2 = scenePath + "debug_super4PCS/pclModel_" + objName + ".ply";
		pcl::io::savePLYFile(input2, *pclModel);

		std::string input3 = scenePath + "debug_super4PCS/pclModelSampled_" + objName + ".ply";
		pcl::io::savePLYFile(input3, *pclModelSampled);

		std::string probImagePath = scenePath + "debug_super4PCS/" + objName + ".png";

		// multi threading
		// th_id = std::thread(getProbableTransformsSuper4PCS, input1, input2, input3, std::ref(bestHypothesis), std::ref(hypothesisSet), probImagePath, PPFMap, max_count_ppf, camIntrinsic, objName);
		getProbableTransformsSuper4PCS(input1, input2, input3, bestHypothesis, hypothesisSet, probImagePath, PPFMap, max_count_ppf, camIntrinsic, objName, scenePath);
		
		std::cout << bestHypothesis.first.matrix() << std::endl;
		
	}

}
