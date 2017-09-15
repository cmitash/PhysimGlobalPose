#ifndef STATE
#define STATE

#include <APCObjects.hpp>
#include <PhySim.hpp>

namespace state{
	class State{
		public:
			State(unsigned int numObjects);
			void expand();
			void copyParent(State*);
			void updateNewObject(apc_objects::APCObjects*, std::pair <Eigen::Isometry3d, float>, int maxDepth);
			void render(Eigen::Matrix4f, std::string);
			void updateStateId(int num);
			void computeCost(cv::Mat obsImg);
			void performICP(std::string scenePath, float max_corr);
			void performTrICP(std::string scenePath, float trimPercentage);
			void correctPhysics(physim::PhySim*, Eigen::Matrix4f, std::string);

			std::string stateId;
			unsigned int numObjects;
			std::vector<std::pair<apc_objects::APCObjects*, Eigen::Isometry3d> > objects;
			float hval;
			unsigned int score;
			cv::Mat renderedImg;
	};

	class Compare{
		public:
			bool operator() (State* lhs, State* rhs){
    			if(lhs->hval != rhs->hval)
    				return (lhs->hval > rhs->hval);
    			else{
    				int ret = lhs->stateId.compare(rhs->stateId);
    				if(ret > 0) return true;
    				else return false;
    			}
			}
	};
}// namespace

#endif