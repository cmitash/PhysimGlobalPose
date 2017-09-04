#ifndef EVALUATE
#define EVALUATE

#include <common_io.h>
#include <Scene.hpp>
#include <APCObjects.hpp>
#include <State.hpp>

namespace evaluate{
	
	class Evaluate{
		public:
			Evaluate();
			void getSuper4pcsError(scene::Scene *currScene, int sceneIdx);
			void readGroundTruth(scene::Scene *currScene);
			void getAllHypoError(scene::Scene *currScene, int sceneIdx);
			void getClusterHypoError(scene::Scene *currScene, int sceneIdx);
			void getSearchResults(scene::Scene *currScene, int sceneIdx);
			void writeResults();

			std::vector<float> rotErr_super4pcs;
			std::vector<float> transErr_super4pcs;
			std::vector<float> emdErr_super4pcs;

			std::vector<float> rotErr_super4pcsICP;
			std::vector<float> transErr_super4pcsICP;
			std::vector<float> emdErr_super4pcsICP;

			std::vector<float> rotErr_allhypothesisMinRot;
			std::vector<float> transErr_allhypothesisMinRot;
			std::vector<float> emdErr_allhypothesisMinRot;

			std::vector<float> rotErr_allhypothesisMintrans;
			std::vector<float> transErr_allhypothesisMintrans;
			std::vector<float> emdErr_allhypothesisMintrans;

			std::vector<float> rotErr_allhypothesisMinEmd;
			std::vector<float> transErr_allhypothesisMinEmd;
			std::vector<float> emdErr_allhypothesisMinEmd;

			std::vector<float> rotErr_clusterhypothesisMinRot;
			std::vector<float> transErr_clusterhypothesisMinRot;
			std::vector<float> emdErr_clusterhypothesisMinRot;

			std::vector<float> rotErr_clusterhypothesisMintrans;
			std::vector<float> transErr_clusterhypothesisMintrans;
			std::vector<float> emdErr_clusterhypothesisMintrans;

			std::vector<float> rotErr_clusterhypothesisMinEmd;
			std::vector<float> transErr_clusterhypothesisMinEmd;
			std::vector<float> emdErr_clusterhypothesisMinEmd;

			std::vector<std::vector<std::pair<float, float> > > rotErr_searchFinal;
			std::vector<std::vector<std::pair<float, float> > > transErr_searchFinal;
			std::vector<std::vector<std::pair<float, float> > > emdErr_searchFinal;
	};
}

#endif