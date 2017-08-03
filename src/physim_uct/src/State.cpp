#include <State.hpp>

void addObjects(pcl::PolygonMesh::Ptr mesh);
void renderDepth(Eigen::Matrix4f pose, cv::Mat &depth_image, std::string path);
void clearScene();

namespace state{
	
	State::State(unsigned int numObjects){
		this->numObjects = numObjects;
		hval = INT_MAX;
		score = INT_MAX;
	}

	void State::expand(){
		std::cout << "***************State::expand***************" << std::endl;
		std::cout << "numObjects: " << numObjects<<std::endl;
		std::cout << "hval: " << hval <<std::endl;
		for(int ii=0; ii<objects.size(); ii++){
			std::cout << "Objects " << ii << ": " << objects[ii].first->objName << std::endl;
		}
	}

	void State::copyParent(State* copyFrom){
		this->objects = copyFrom->objects;
		this->stateId = copyFrom->stateId;
	}

	void State::render(Eigen::Matrix4f cam_pose, std::string scenePath, cv::Mat &depth_image){
		clearScene();
		for(int ii=0; ii<objects.size(); ii++){
  			pcl::PolygonMesh::Ptr mesh_in (new pcl::PolygonMesh (objects[ii].first->objModel));
  			pcl::PolygonMesh::Ptr mesh_out (new pcl::PolygonMesh (objects[ii].first->objModel));
  			Eigen::Matrix4f transform;
  			utilities::convertToMatrix(objects[ii].second, transform);
  			utilities::convertToWorld(transform, cam_pose);
  			utilities::TransformPolyMesh(mesh_in, mesh_out, transform);
  			addObjects(mesh_out);
		}
		renderDepth(cam_pose, depth_image, scenePath + "/debug/render" + stateId + ".png");
	}

	void State::updateNewObject(apc_objects::APCObjects* newObj, std::pair <Eigen::Isometry3d, float> pose, int maxDepth){
		objects.push_back(std::make_pair(newObj, pose.first));
		hval = (1 - pose.second)*(maxDepth - numObjects);
	}

	void State::updateStateId(int num){
		char nums[20];
		sprintf(nums,"_%d", num);
		stateId.append(nums);
	}

	void State::computeCost(cv::Mat renderedImg, cv::Mat obsImg){
		renderedImg.convertTo(renderedImg, CV_32FC1);
		renderedImg = renderedImg/1000;
		float obScore = 0;
		float renScore = 0;
		float intScore = 0;

	    for (int i = 0; i < obsImg.rows; ++i)
	    {
	        float* pObs = obsImg.ptr<float>(i);
	        float* pRen = renderedImg.ptr<float>(i);
	        for (int j = 0; j < obsImg.cols; ++j)
	        {
	            float obVal = *pObs++;
	            float renVal = *pRen++;

	            float absDiff = abs(obVal - renVal);

	            if(obVal)obScore += absDiff;
	            if(renVal < 1)renScore += absDiff;
	            if(obVal && (renVal < 1))intScore += absDiff;
	        }
	    }
	    score = obScore + renScore - intScore;
	    std::cout<<"score: "<<score<<std::endl;
	}

} // namespace