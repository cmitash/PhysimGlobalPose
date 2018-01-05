#include <GlobalCfg.hpp>
#include <SceneCfg.hpp>
#include <PoseVisualization.hpp>

// global config pointer
GlobalCfg *pCfg;

/********************************* function: main *******************************************************
********************************************************************************************************/

int main(int argc, char **argv){
  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
  ros::init(argc, argv, "physim_node");
  
  pCfg = new GlobalCfg();

  pCfg->loadObjects();

  if (argc < 3){
  	std::cout << "Enter the scene location and dataset type to run this tool !!!!" << std::endl;
  	exit(-1);
  }

  std::string OperationMode = argv[1];
  std::string sceneFile = argv[2];

  // Initialize the scene based on the type of dataset or camera input is chosen as default
  scene_cfg::SceneCfg *currScene;
  if(!OperationMode.compare("APC"))
    currScene = new scene_cfg::APCSceneCfg(sceneFile, "DUMMY", "DUMMY", "DUMMY");
  else if(!OperationMode.compare("YCB"))
    currScene = new scene_cfg::YCBSceneCfg(sceneFile, "DUMMY", "DUMMY", "DUMMY");
  else
    currScene = new scene_cfg::CAMSceneCfg(sceneFile, "DUMMY", "DUMMY", "DUMMY");

  currScene->getSceneInfo(pCfg);

  std::cout<<"number of objects: " << currScene->numObjects << std::endl
           <<"camera pose: " << std::endl << currScene->camPose << std::endl
           <<"camera intrinsics: "<< std::endl << currScene->camIntrinsic << std::endl;
  
  pose_visualization::PoseVisualization *pViz = new pose_visualization::PoseVisualization();
  pViz->loadSceneCloud(currScene->depthImage, currScene->colorImage, currScene->camIntrinsic, currScene->camPose);
 
  ifstream pFile;
  pFile.open ((currScene->scenePath + "result.txt").c_str());
  for (int ii=0; ii<currScene->numObjects; ii++){
  	char objName[30];
  	std::vector<double> v(7,0);
    pFile >> objName >> v[0] >> v[1] >> v[2] >> v[3] >> v[4] >> v[5] >> v[6];
  	pViz->loadObjectModels(currScene->pSceneObjects[ii]->pObject->objModel, v, std::string(objName));
  }
  pFile.close();
  pViz->startViz();

  return 0;
}

/********************************* end of functions ****************************************************
*******************************************************************************************************/