#ifndef POSE_CANDIDATES
#define POSE_CANDIDATES

#include <common_io.h>

namespace pose_candidates{
	
	class ObjectPoseCandidateSet{
	public:
		ObjectPoseCandidateSet();
		~ObjectPoseCandidateSet();

		void generate(std::string objName, std::string scenePath, 
				PointCloudRGB::Ptr pclSegment, PointCloudRGB::Ptr pclModel);
		void cluster();

		std::vector< std::pair <Eigen::Isometry3d, float> > hypothesisSet;
		std::pair <Eigen::Isometry3d, float> bestHypothesis;
	};
}

#endif