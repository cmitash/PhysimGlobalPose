#include <Scene.hpp>
#include <fstream>

#include <detection_package/UpdateActiveListFrame.h>
#include <detection_package/UpdateBbox.h>

namespace scene{
	Scene::Scene(std::string scenePath){
		// Read objects file and add the scene objects
		std::ifstream filein(scenePath + "objects.txt");
		std::string line;
		while(std::getline(filein, line)){
			for(int i=0;i<Objects.size();i++){
				if(!line.compare(Objects[i]->objName))
					sceneObjs.push_back(Objects[i]);
			}
		}
		this->scenePath = scenePath;
		numObjects = sceneObjs.size();
		filein.close();

		// Read camera pose
		camPose = Eigen::Matrix4f::Zero(4,4);
		filein.open (scenePath + "cameraExtrinsic.txt", std::ifstream::in);
		for(int i=0; i<4; i++)
			for(int j=0; j<4; j++)
	  			filein >> camPose(i,j);
	  	filein.close();

	  	// Read camera intrinsic matrix
		camIntrinsic = Eigen::Matrix3f::Zero(3,3);
		filein.open (scenePath + "cameraIntinsic.txt", std::ifstream::in);
		for(int i=0; i<3; i++)
			for(int j=0; j<3; j++)
	  			filein >> camIntrinsic(i,j);
	  	filein.close();

	  	color_image = cv::imread(scenePath + "frame-000000.color.png", CV_LOAD_IMAGE_COLOR);
	    depth_image = cv::imread(scenePath + "frame-000000.depth.png", CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
	}

	void Scene::performRCNNDetection(){
		ros::NodeHandle n;
		ros::ServiceClient clientlist = n.serviceClient<detection_package::UpdateActiveListFrame>("/update_active_list_and_frame");
		ros::ServiceClient clientbox = n.serviceClient<detection_package::UpdateBbox>("/update_bbox");
		detection_package::UpdateActiveListFrame listsrv;
		detection_package::UpdateBbox boxsrv;

		// Update object list
		for(int i=0;i<numObjects;i++){
			listsrv.request.active_list.push_back(sceneObjs[i]->objIdx);
			listsrv.request.active_frame = "000000";
		    if(clientlist.call(listsrv))
    			ROS_INFO("Returned : %d", (bool)listsrv.response.result);
  		  	else{
    			ROS_ERROR("Failed to call service UpdateActiveListFrame");
    			exit(1);
    	  	}
    	}

    	// Call R-CNN
    	boxsrv.request.scene_path = scenePath;
    	if (clientbox.call(boxsrv))
      		for(int i=0;i<numObjects;i++)
        		sceneObjs[i]->bbox	= cv::Rect(boxsrv.response.tl_x[i], boxsrv.response.tl_y[i], boxsrv.response.br_x[i] - boxsrv.response.tl_x[i],
                    							boxsrv.response.br_y[i] - boxsrv.response.tl_y[i]);
        else{
      		ROS_ERROR("Failed to call service UpdateBbox");
      		exit(1);
    	}
	}
}