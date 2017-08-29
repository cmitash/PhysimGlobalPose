#ifndef SEARCH
#define SEARCH

#include <State.hpp>
#include <Scene.hpp>
#include <PhySim.hpp>

namespace search{
	
	class Search{
		public:
			Search(std::vector<apc_objects::APCObjects*> objOrder, 
					std::vector< std::vector< std::pair <Eigen::Isometry3d, float> > > unconditionedHypothesis,
					std::string scenePath, Eigen::Matrix4f camPose, cv::Mat depthImage, std::vector<float> cutOffScore);
			void heuristicSearch();
			void expandNode(state::State*);

			state::State *rootState;
			
			std::vector<apc_objects::APCObjects*> objOrder;
			std::vector< std::vector< std::pair <Eigen::Isometry3d, float> > > unconditionedHypothesis;
			std::string scenePath;
			Eigen::Matrix4f camPose;
			cv::Mat depthImage;
			std::vector<float> cutOffScore;

			state::State *bestState;
			unsigned int bestScore;

			physim::PhySim *pSim;
			std::priority_queue<state::State*> pq;

	};
}// namespace

#endif