#ifndef HYPOTHESIS_SELECTION
#define HYPOTHESIS_SELECTION

#include <common_io.h>
#include <SceneCfg.hpp>
#include <mcts/UCTSearch.hpp>

namespace hypothesis_selection{
	
	class HypothesisSelection{
	public:
		HypothesisSelection();
		~HypothesisSelection();

		virtual void selectBestPoses(scene_cfg::SceneCfg *pCfg){}
		void greedyClustering(scene_cfg::SceneCfg *pCfg, int objId);
	};

	class LCPSelection: public HypothesisSelection{

		void selectBestPoses(scene_cfg::SceneCfg *pCfg);
	};

	class MCTSSelection: public HypothesisSelection{

		void selectBestPoses(scene_cfg::SceneCfg *pCfg);
	};
}

#endif