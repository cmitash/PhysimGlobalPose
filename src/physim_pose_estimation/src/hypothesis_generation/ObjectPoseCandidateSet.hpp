#ifndef POSE_CANDIDATES
#define POSE_CANDIDATES

#include <common_io.h>

namespace pose_candidates{
	
	class ObjectPoseCandidateSet{
	public:
		ObjectPoseCandidateSet();
		~ObjectPoseCandidateSet();

		void generate(std::string objName, std::string scenePath, 
				pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclSegment, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclModel, 
				pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclModelSampled, std::map<std::vector<int>, std::vector<std::pair<int,int> > > &PPFMap, 
				int max_count_ppf, Eigen::Matrix4f camPose, Eigen::Matrix3f camIntrinsic);
		void cluster();

		std::vector< std::pair <Eigen::Isometry3d, float> > hypothesisSet;
		std::pair <Eigen::Isometry3d, float> bestHypothesis;
	};
}

#endif