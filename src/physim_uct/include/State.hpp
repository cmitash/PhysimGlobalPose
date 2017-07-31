#ifndef STATE
#define STATE

#include <APCObjects.hpp>

namespace state{
	
	class State{
		public:
			State(unsigned int numObjects);

			unsigned int numObjects;
			std::vector<std::pair<apc_objects::APCObjects*, geometry_msgs::Pose> > objects;
			unsigned int score;
			std::vector<State*> children;
			State* parent;
	};
}// namespace

#endif