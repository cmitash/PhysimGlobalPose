#ifndef HYPOTHESIS_SELECTION
#define HYPOTHESIS_SELECTION

#include <common_io.h>
#include <SceneCfg.hpp>

namespace hypothesis_selection{
	
	class HypothesisSelection{
	public:
		HypothesisSelection();
		~HypothesisSelection();

		virtual void selectBestPoses(scene_cfg::SceneCfg *pCfg){}
	};

	class LCPSelection: public HypothesisSelection{

		void selectBestPoses(scene_cfg::SceneCfg *pCfg);
	};
}

#endif