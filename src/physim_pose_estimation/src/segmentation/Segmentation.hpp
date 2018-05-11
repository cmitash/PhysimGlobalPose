#ifndef SEGMENTATION
#define SEGMENTATION

#include <common_io.h>
#include <GlobalCfg.hpp>
#include <SceneCfg.hpp>

namespace segmentation{
	
	class Segmentation{
	public:
		Segmentation();
		~Segmentation();
		void compute3dSegment(scene_cfg::SceneCfg *sCfg);
		virtual void compute2dSegment(GlobalCfg *gCfg, scene_cfg::SceneCfg *sCfg){}
	};

	class RCNNSegmentation: public Segmentation {
	public:
		void compute2dSegment(GlobalCfg *gCfg, scene_cfg::SceneCfg *sCfg);
	};

	class RCNNThresholdSegmentation: public Segmentation {
	public:
		void compute2dSegment(GlobalCfg *gCfg, scene_cfg::SceneCfg *sCfg);
	};

	class FCNSegmentation: public Segmentation {
	public:
		void compute2dSegment(GlobalCfg *gCfg, scene_cfg::SceneCfg *sCfg);
	};

	class FCNThresholdSegmentation: public Segmentation {
	public:
		void compute2dSegment(GlobalCfg *gCfg, scene_cfg::SceneCfg *sCfg);
	};

	class GTSegmentation: public Segmentation {
	public:
		void compute2dSegment(GlobalCfg *gCfg, scene_cfg::SceneCfg *sCfg);
	};
}

#endif

