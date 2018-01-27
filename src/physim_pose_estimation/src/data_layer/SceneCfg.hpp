#ifndef SCENE_CFG
#define SCENE_CFG

#include <GlobalCfg.hpp>
#include <ObjectPoseCandidateSet.hpp>
#include <Objects.hpp>

namespace scene_cfg{
	
	class SceneObjects{
		public:
			objects::Objects *pObject;
			cv::Mat objMask;
			pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclSegment;
			pose_candidates::ObjectPoseCandidateSet *hypotheses;
			Eigen::Isometry3d objPose;

			SceneObjects(){}
	};

	class SceneCfg{
		public:
			SceneCfg(std::string SceneFiles, std::string SegmentationMode, 
						std::string HypothesisGenerationMode, std::string HypothesisVerificationMode);
			~SceneCfg();

			void removeTable();
			void getTableParams();
			void perfromSegmentation(GlobalCfg *pCfg);
			void generateHypothesis();
			void performHypothesisSelection();

			virtual void getSceneInfo(GlobalCfg *pCfg){}
			virtual void cleanDebugLocations(){}

			int numObjects;
			std::vector<SceneObjects*> pSceneObjects;

			std::string scenePath;

			cv::Mat colorImage;
			cv::Mat depthImage;
			PointCloudRGB::Ptr sceneCloud;

			Eigen::Matrix4f camPose;
			Eigen::Matrix3f camIntrinsic;
			std::vector< float> tableParams;

			std::string segMode;
			std::string hypoGenMode;
			std::string HVMode;
	};

	class APCSceneCfg : public SceneCfg{
		public:
			APCSceneCfg(std::string SceneFiles, std::string SegmentationMode, 
						std::string HypothesisGenerationMode, std::string HypothesisVerificationMode) : 
						SceneCfg(SceneFiles, SegmentationMode, HypothesisGenerationMode, HypothesisVerificationMode){};

			void getSceneInfo(GlobalCfg *pCfg);
			void cleanDebugLocations();

	};

	class YCBSceneCfg : public SceneCfg{
		public:
			YCBSceneCfg(std::string SceneFiles, std::string SegmentationMode, 
						std::string HypothesisGenerationMode, std::string HypothesisVerificationMode) : 
						SceneCfg(SceneFiles, SegmentationMode, HypothesisGenerationMode, HypothesisVerificationMode){};
			
			void getSceneInfo(GlobalCfg *pCfg);
			void cleanDebugLocations();

	};

	class CAMSceneCfg : public SceneCfg{
		public:
			CAMSceneCfg(std::string SceneFiles, std::string SegmentationMode, 
						std::string HypothesisGenerationMode, std::string HypothesisVerificationMode) :
						SceneCfg(SceneFiles, SegmentationMode, HypothesisGenerationMode, HypothesisVerificationMode){};

			void getSceneInfo(GlobalCfg *pCfg);

			void cleanDebugLocations();
	};

}//namespace
#endif