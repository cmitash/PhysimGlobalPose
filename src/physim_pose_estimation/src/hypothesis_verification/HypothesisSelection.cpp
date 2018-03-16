#include <HypothesisSelection.hpp>

namespace hypothesis_selection{

	HypothesisSelection::HypothesisSelection(){

	}

	HypothesisSelection::~HypothesisSelection(){

	}

	void LCPSelection::selectBestPoses(scene_cfg::SceneCfg *pCfg){

		for(int ii=0; ii<pCfg->numObjects; ii++){
			pCfg->pSceneObjects[ii]->objPose = pCfg->pSceneObjects[ii]->hypotheses->bestHypothesis.first;
			// utilities::performTrICP(pCfg->pSceneObjects[ii]->pclSegment, pCfg->pSceneObjects[ii]->pObject->pclModel, 
			// 				pCfg->pSceneObjects[ii]->objPose, pCfg->pSceneObjects[ii]->objPose, 0.9);
			std::cout << "hypothesis size: " << pCfg->pSceneObjects[ii]->hypotheses->hypothesisSet.size() << std::endl;
			int jj=0;
			for(auto it: pCfg->pSceneObjects[ii]->hypotheses->hypothesisSet) {
				pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZRGBNormal>);;
				pcl::transformPointCloud(*pCfg->pSceneObjects[ii]->pObject->pclModel, *cloud_out, it.first.matrix());
				char hypFile[200];
				sprintf(hypFile, "debug_super4PCS/hypothesis_%d_%d.ply", ii, jj);
				pcl::io::savePLYFile((pCfg->scenePath + hypFile).c_str(), *cloud_out);
				jj++;
			}
		}
	}
}