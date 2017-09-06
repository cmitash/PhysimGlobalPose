#include <UCTState.hpp>

void addObjects(pcl::PolygonMesh::Ptr mesh);
void renderDepth(Eigen::Matrix4f pose, cv::Mat &depth_image, std::string path);
void clearScene();

namespace uct_state{
	float explanationThreshold = 0.005;
	float alpha = 0.5;

	/********************************* function: constructor ***********************************************
	*******************************************************************************************************/

	UCTState::UCTState(unsigned int numObjects, 
			std::vector< std::pair <Eigen::Isometry3d, float> > hypSet, UCTState* parent) : isExpanded(hypSet.size(), 0){
		this->numObjects = numObjects;
		hval = 0;
		score = INT_MAX;
		numChildren = hypSet.size();
		parentState = parent;
	}

	/********************************* function: copyParent ************************************************
	*******************************************************************************************************/

	void UCTState::copyParent(UCTState* copyFrom){
		this->objects = copyFrom->objects;
		this->stateId = copyFrom->stateId;
	}

	/********************************* function: render ****************************************************
	*******************************************************************************************************/

	void UCTState::render(Eigen::Matrix4f cam_pose, std::string scenePath, cv::Mat &depth_image){
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
		renderDepth(cam_pose, depth_image, scenePath + "debug/render" + stateId + ".png");
	}

	/********************************* function: updateNewObject *******************************************
	*******************************************************************************************************/

	void UCTState::updateNewObject(apc_objects::APCObjects* newObj, std::pair <Eigen::Isometry3d, float> pose, int maxDepth){
		objects.push_back(std::make_pair(newObj, pose.first));
	}

	/********************************* function: updateStateId *********************************************
	*******************************************************************************************************/

	void UCTState::updateStateId(int num){
		char nums[20];
		sprintf(nums,"_%d", num);
		stateId.append(nums);
	}

	/********************************* function: computeCost ***********************************************
	*******************************************************************************************************/

	void UCTState::computeCost(cv::Mat renderedImg, cv::Mat obsImg){
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

	            float absDiff = fabs(obVal - renVal);

	            if(obVal > 0 && absDiff < explanationThreshold)obScore++;
	            if(renVal > 0 && absDiff < explanationThreshold)renScore++;
	            if(obVal > 0 && renVal > 0 && absDiff < explanationThreshold)intScore++;
	        }
	    }
	    score = obScore + renScore - intScore;
	    std::cout<<"score: "<<score<<std::endl;
	}

	/********************************* function: performICP ************************************************
	*******************************************************************************************************/

	void UCTState::performICP(std::string scenePath, float max_corr){
		if(!numObjects)
			return;
		PointCloud::Ptr transformedCloud (new PointCloud);
		PointCloud icptransformedCloud;
		Eigen::Matrix4f tform;
		utilities::convertToMatrix(objects[numObjects-1].second, tform);
		pcl::transformPointCloud(*objects[numObjects-1].first->pclModel, *transformedCloud, tform);

		pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
		icp.setInputSource(transformedCloud);
		icp.setInputTarget(objects[numObjects-1].first->pclSegment);
		icp.setMaxCorrespondenceDistance (max_corr);
		icp.setMaximumIterations (50);
		icp.setTransformationEpsilon (1e-8);
		icp.align(icptransformedCloud);

		#ifdef DBG_ICP
		std::string input2 = scenePath + "debug/render" + stateId + "_segment.ply";
		pcl::io::savePLYFile(input2, *objects[numObjects-1].first->pclSegment);

		pcl::transformPointCloud(*objects[numObjects-1].first->pclModel, *transformedCloud, tform);
		std::string input1 = scenePath + "debug/render" + stateId + "_Premodel.ply";
		pcl::io::savePLYFile(input1, *transformedCloud);
		#endif

		tform = icp.getFinalTransformation()*tform;
		
		#ifdef DBG_ICP
		pcl::transformPointCloud(*objects[numObjects-1].first->pclModel, *transformedCloud, tform);
		std::string input3 = scenePath + "debug/render" + stateId + "_Postmodel.ply";
		pcl::io::savePLYFile(input3, *transformedCloud);
		#endif

		utilities::convertToIsometry3d(tform, objects[numObjects-1].second);
	}

	/********************************* function: performTrICP **********************************************
	*******************************************************************************************************/

	void UCTState::performTrICP(std::string scenePath, float trimPercentage){
		if(!numObjects)
			return;
		PointCloud::Ptr transformedCloud (new PointCloud);
		PointCloud::Ptr unexplainedSegment (new PointCloud);
		PointCloud::Ptr explainedPts (new PointCloud);
		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
		Eigen::Matrix4f tform;
		
		// initialize trimmed ICP
		pcl::recognition::TrimmedICP<pcl::PointXYZ, float> tricp;
		tricp.init(objects[numObjects-1].first->pclModel);
		tricp.setNewToOldEnergyRatio(1.f);

		// remove points explained by objects already placed objects
		copyPointCloud(*objects[numObjects-1].first->pclSegment, *unexplainedSegment);
		
		#ifdef DBG_ICP
		std::string input1 = scenePath + "debug/render" + stateId + "_Presegment.ply";
		pcl::io::savePLYFile(input1, *unexplainedSegment);
		#endif

		if(numObjects > 1){
			for(int ii=0; ii<numObjects-1; ii++) {
				Eigen::Matrix4f objPose;
				utilities::convertToMatrix(objects[ii].second, objPose);
				pcl::transformPointCloud(*objects[ii].first->pclModel, *transformedCloud, objPose);
				*explainedPts += *transformedCloud;
			}

			kdtree.setInputCloud (explainedPts);

			for(int ii=0; ii<unexplainedSegment->points.size(); ii++){
				std::vector<int> pointIdxRadiusSearch;
	  			std::vector<float> pointRadiusSquaredDistance;
	  			if (kdtree.radiusSearch (unexplainedSegment->points[ii], 0.005, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 ){
	  				unexplainedSegment->points.erase(unexplainedSegment->points.begin() + ii);
	  				unexplainedSegment->width--;
	  			}
			}
		}

		#ifdef DBG_ICP
		std::string input2 = scenePath + "debug/render" + stateId + "_Postsegment.ply";
		pcl::io::savePLYFile(input2, *unexplainedSegment);
		#endif

		float numPoints = trimPercentage*unexplainedSegment->points.size();
		
		// get current object transform
		utilities::convertToMatrix(objects[numObjects-1].second, tform);
		tform = tform.inverse().eval();

		#ifdef DBG_ICP
		std::cout<< "size of cloud: "<<abs(numPoints)<<std::endl;
		pcl::transformPointCloud(*objects[numObjects-1].first->pclModel, *transformedCloud, tform.inverse().eval());
		std::string input3 = scenePath + "debug/render" + stateId + "_Premodel.ply";
		pcl::io::savePLYFile(input3, *transformedCloud);
		#endif

		tricp.align(*unexplainedSegment, abs(numPoints), tform);

		tform = tform.inverse().eval();

		#ifdef DBG_ICP
		pcl::transformPointCloud(*objects[numObjects-1].first->pclModel, *transformedCloud, tform);
		std::string input4 = scenePath + "debug/render" + stateId + "_Postmodel.ply";
		pcl::io::savePLYFile(input4, *transformedCloud);
		#endif

		utilities::convertToIsometry3d(tform, objects[numObjects-1].second);
	}

	/********************************* function: correctPhysics ********************************************
	*******************************************************************************************************/
	void UCTState::correctPhysics(physim::PhySim* pSim, Eigen::Matrix4f cam_pose, std::string scenePath){
		if(!numObjects)
			return;

		for(int ii=0; ii<numObjects-1; ii++){
			Eigen::Matrix4f camTform, worldTform;
			Eigen::Isometry3d worldPose;
			utilities::convertToMatrix(objects[ii].second, camTform);
			utilities::convertToWorld(camTform, cam_pose);
			utilities::convertToIsometry3d(camTform, worldPose);
			pSim->addObject(objects[ii].first->objName, worldPose, 0.0f);
		}

		Eigen::Matrix4f camTform, worldTform;
		Eigen::Isometry3d worldPose;
		utilities::convertToMatrix(objects[numObjects-1].second, camTform);
		utilities::convertToWorld(camTform, cam_pose);
		utilities::convertToIsometry3d(camTform, worldPose);

		pSim->addObject(objects[numObjects-1].first->objName, worldPose, 10.0f);

		#ifdef DBG_PHYSICS
		std::ofstream cfg_in;
		std::string path_in = scenePath + "debug/render" + stateId + "_in.txt";
		cfg_in.open (path_in.c_str(), std::ofstream::out | std::ofstream::app);
		Eigen::Isometry3d pose_in;
		for(int ii=0; ii<numObjects; ii++){
			pSim->getTransform(objects[ii].first->objName, pose_in);
			Eigen::Vector3d trans_in = pose_in.translation();
			Eigen::Quaterniond rot_in(pose_in.rotation()); 
			cfg_in << objects[ii].first->objName << " " << trans_in[0] << " " << trans_in[1] << " " << trans_in[2]
				<< " " << rot_in.x() << " " << rot_in.y() << " " << rot_in.z() << " " << rot_in.w() << std::endl;
		}
		cfg_in.close();
		#endif

		pSim->simulate(60);

		#ifdef DBG_PHYSICS
		std::ofstream cfg_out;
		std::string path_out = scenePath + "debug/render" + stateId + "_out.txt";
		cfg_out.open (path_out.c_str(), std::ofstream::out | std::ofstream::app);
		Eigen::Isometry3d pose_out;
		for(int ii=0; ii<numObjects; ii++){
			pSim->getTransform(objects[ii].first->objName, pose_out);
			Eigen::Vector3d trans_out = pose_out.translation();
			Eigen::Quaterniond rot_out(pose_out.rotation()); 
			cfg_out << objects[ii].first->objName << " " << trans_out[0] << " " << trans_out[1] << " " << trans_out[2]
				<< " " << rot_out.x() << " " << rot_out.y() << " " << rot_out.z() << " " << rot_out.w() << std::endl;
		}
		cfg_out.close();
		#endif

		Eigen::Isometry3d finalPose;
		Eigen::Matrix4f tform;
		pSim->getTransform(objects[numObjects-1].first->objName, finalPose);
		utilities::convertToMatrix(finalPose, tform);
		utilities::convertToCamera(tform, cam_pose);
		utilities::convertToIsometry3d(tform, objects[numObjects-1].second);

		for(int ii=0; ii<numObjects; ii++)
			pSim->removeObject(objects[ii].first->objName);
	}

	/******************************** function: getBestChild ************************************************
	/*******************************************************************************************************/

	uct_state::UCTState* UCTState::getBestChild(){
		int bestChildIdx = -1;
		float bestVal = 0;

		for(int ii=0; ii<numChildren; ii++){
			float tmpVal = (children[ii]->hval/children[ii]->numExpansions) + alpha*sqrt(2*log(numExpansions)/children[ii]->numExpansions);
			if (tmpVal > bestVal){
				bestVal = tmpVal;
				bestChildIdx = ii;
			}
		}

		return children[bestChildIdx];
	}

	/******************************** function: isFullyExpanded *********************************************
	/*******************************************************************************************************/

	bool UCTState::isFullyExpanded(){
		for(int ii=0; ii<numChildren; ii++)
			if(isExpanded[ii] == 0)return false;

		return true;
	}

	/********************************* end of functions ****************************************************
	*******************************************************************************************************/

} // namespace