#include <common_io.h>

int numBinsEMD = 20;

namespace utilities{
	
	/********************************* function: meshgrid **************************************************
	Reference: http://answers.opencv.org/question/11788/is-there-a-meshgrid-function-in-opencv
	*******************************************************************************************************/

	static void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat1i &X, cv::Mat1i &Y){
		std::vector<int> t_x, t_y;
		for (int i = xgv.start; i <= xgv.end; i++) t_x.push_back(i);
		for (int i = ygv.start; i <= ygv.end; i++) t_y.push_back(i);
		cv::repeat(cv::Mat(t_x).reshape(1,1), cv::Mat(t_y).total(), 1, X);
		cv::repeat(cv::Mat(t_y).reshape(1,1).t(), 1, cv::Mat(t_x).total(), Y);
	}

	/********************************* function: type2str **************************************************
	Reference: https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-
	with-mattype-in-opencv
	*******************************************************************************************************/

	std::string type2str(int type){
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);
	switch (depth){
		case CV_8U:  r = "8U"; break;
		case CV_8S:  r = "8S"; break;
		case CV_16U: r = "16U"; break;
		case CV_16S: r = "16S"; break;
		case CV_32S: r = "32S"; break;
		case CV_32F: r = "32F"; break;
		case CV_64F: r = "64F"; break;
		default:     r = "User"; break;
	}
    r += "C";
    r += (chans+'0');
	return r;
	}

	/********************************* function: readDepthImage ********************************************
	*******************************************************************************************************/

	void readDepthImage(cv::Mat &depthImg, std::string path){
		std::cout << path << std::endl;
		cv::Mat depthImgRaw = cv::imread(path, CV_16UC1);
		depthImg = cv::Mat::zeros(depthImgRaw.rows, depthImgRaw.cols, CV_32FC1);
		for(int u=0; u<depthImgRaw.rows; u++)
			for(int v=0; v<depthImgRaw.cols; v++){
				unsigned short depthShort = depthImgRaw.at<unsigned short>(u,v);

				//TODO: need to manually uncomment for APC objects
				depthShort = (depthShort << 13 | depthShort >> 3);
				
				float depth = (float)depthShort/10000;
				depthImg.at<float>(u, v) = depth;
			}
	}

	/********************************* function: readProbImage ********************************************
	*******************************************************************************************************/

	void readProbImage(cv::Mat &probImg, std::string path){
		cv::Mat depthImgRaw = cv::imread(path, CV_16UC1);
		probImg = cv::Mat::zeros(depthImgRaw.rows, depthImgRaw.cols, CV_32FC1);
		for(int u=0; u<depthImgRaw.rows; u++)
			for(int v=0; v<depthImgRaw.cols; v++){
				unsigned short depthShort = depthImgRaw.at<unsigned short>(u,v);
				
				float depth = (float)depthShort/10000;
				probImg.at<float>(u, v) = depth;
			}
	}
	
	/********************************* function: writeDepthImage *******************************************
	*******************************************************************************************************/

	void writeDepthImage(cv::Mat &depthImg, std::string path){
		cv::Mat depthImgRaw = cv::Mat::zeros(depthImg.rows, depthImg.cols, CV_16UC1);
		for(int u=0; u<depthImg.rows; u++)
			for(int v=0; v<depthImg.cols; v++){
				float depth = depthImg.at<float>(u,v)*10000;
				unsigned short depthShort = (unsigned short)depth;
				// depthShort = (depthShort << 3 | depthShort >> 13);
				depthImgRaw.at<unsigned short>(u, v) = depthShort;
			}
		cv::imwrite(path, depthImgRaw);
	}

	/********************************* function: writeClassImage *******************************************
	*******************************************************************************************************/

	void writeClassImage(cv::Mat &classImg, cv::Mat colorImage, std::string path){
		cv::Mat classImgRaw(classImg.rows, classImg.cols, CV_8UC3, cv::Scalar(0, 0, 0));
		cv::Mat colArray(4,1,CV_8UC3, cv::Scalar(0, 0, 0));
		colArray.at<cv::Vec3b>(0,0)[0] = 255;
		colArray.at<cv::Vec3b>(1,0)[1] = 255;
		colArray.at<cv::Vec3b>(2,0)[2] = 255;

		for(int u=0; u<classImg.rows; u++)
			for(int v=0; v<classImg.cols; v++){
				int classVal = classImg.at<uchar>(u,v);
				if (classVal > 0){
					classImgRaw.at<cv::Vec3b>(u,v)[0] = colArray.at<cv::Vec3b>(classVal-1,0)[0];
					classImgRaw.at<cv::Vec3b>(u,v)[1] = colArray.at<cv::Vec3b>(classVal-1,0)[1];
					classImgRaw.at<cv::Vec3b>(u,v)[2] = colArray.at<cv::Vec3b>(classVal-1,0)[2];
				}
			}

		cv::Mat vizImage(classImg.rows, classImg.cols, CV_8UC3, cv::Scalar(0, 0, 0));
		double alpha = 0.6;
		double beta = ( 1.0 - alpha );
		addWeighted(colorImage, alpha, classImgRaw, beta, 0.0, vizImage);
		cv::imwrite(path, vizImage);
	}

	/********************************* function: convert3dOrganized ****************************************
	Description: Convert Depth image to point cloud. TODO: Could it be faster?
	Reference: https://gist.github.com/jacyzon/fa868d0bcb13abe5ade0df084618cf9c
	*******************************************************************************************************/

	void convert3dOrganized(cv::Mat &objDepth, Eigen::Matrix3f &camIntrinsic, PointCloud::Ptr objCloud){
		int imgWidth = objDepth.cols;
		int imgHeight = objDepth.rows;
		
		objCloud->height = (uint32_t) imgHeight;
		objCloud->width = (uint32_t) imgWidth;
		objCloud->is_dense = false;
		objCloud->points.resize(objCloud->width * objCloud->height);

		// Try meshgrid implementation
		// cv::Mat1i X, Y;
		// meshgrid(cv::Range(1,imgWidth), cv::Range(1, imgHeight), X, Y);
		// printf("Matrix: %s %dx%d \n", utilities::type2str( depth_image.type() ).c_str(), depth_image.cols, depth_image.rows );
		// cv::Mat CamX = ((X-camIntrinsic(0,2)).mul(objDepth))/camIntrinsic(0,0);
		// cv::Mat CamY = ((Y-camIntrinsic(1,2)).mul(objDepth))/camIntrinsic(1,1);

		for(int u=0; u<imgHeight; u++)
			for(int v=0; v<imgWidth; v++){
				float depth = objDepth.at<float>(u,v);
				if(depth > 0.1 && depth < 2.0){
					objCloud->at(v, u).x = (float)((v - camIntrinsic(0,2)) * depth / camIntrinsic(0,0));
					objCloud->at(v, u).y = (float)((u - camIntrinsic(1,2)) * depth / camIntrinsic(1,1));
					objCloud->at(v, u).z = depth;
				}
		}
	}

	/********************************* function: convert3dOrganizedRGB *************************************
	Description: Convert Depth image to point cloud. TODO: Could it be faster?
	Reference: https://gist.github.com/jacyzon/fa868d0bcb13abe5ade0df084618cf9c
	*******************************************************************************************************/

	void convert3dOrganizedRGB(cv::Mat &objDepth, cv::Mat &colImage, Eigen::Matrix3f &camIntrinsic, PointCloudRGB::Ptr objCloud){
		int imgWidth = objDepth.cols;
		int imgHeight = objDepth.rows;
		
		objCloud->height = (uint32_t) imgHeight;
		objCloud->width = (uint32_t) imgWidth;
		objCloud->is_dense = false;
		objCloud->points.resize(objCloud->width * objCloud->height);

		// Try meshgrid implementation
		// cv::Mat1i X, Y;
		// meshgrid(cv::Range(1,imgWidth), cv::Range(1, imgHeight), X, Y);
		// printf("Matrix: %s %dx%d \n", utilities::type2str( depth_image.type() ).c_str(), depth_image.cols, depth_image.rows );
		// cv::Mat CamX = ((X-camIntrinsic(0,2)).mul(objDepth))/camIntrinsic(0,0);
		// cv::Mat CamY = ((Y-camIntrinsic(1,2)).mul(objDepth))/camIntrinsic(1,1);

		for(int u=0; u<imgHeight; u++)
			for(int v=0; v<imgWidth; v++){
				float depth = objDepth.at<float>(u,v);
				cv::Vec3b colour = colImage.at<cv::Vec3b>(u,v);
				if(depth > 0.1 && depth < 2.0){
					objCloud->at(v, u).x = (float)((v - camIntrinsic(0,2)) * depth / camIntrinsic(0,0));
					objCloud->at(v, u).y = (float)((u - camIntrinsic(1,2)) * depth / camIntrinsic(1,1));
					objCloud->at(v, u).z = depth;
					uint32_t rgb = ((uint32_t)colour.val[2] << 16 | (uint32_t)colour.val[1] << 8 | (uint32_t)colour.val[0]);
					objCloud->at(v, u).rgb = *reinterpret_cast<float*>(&rgb);
				}
		}
	}

	/********************************* function: convert3dUnOrganized **************************************
	*******************************************************************************************************/

	void convert3dUnOrganized(cv::Mat &objDepth, Eigen::Matrix3f &camIntrinsic, PointCloud::Ptr objCloud){
		int imgWidth = objDepth.cols;
		int imgHeight = objDepth.rows;
		
		for(int u=0; u<imgHeight; u++)
			for(int v=0; v<imgWidth; v++){
				float depth = objDepth.at<float>(u,v);
				if(depth > 0.1 && depth < 2.0){
					pcl::PointXYZ pt;
					pt.x = (float)((v - camIntrinsic(0,2)) * depth / camIntrinsic(0,0));
					pt.y = (float)((u - camIntrinsic(1,2)) * depth / camIntrinsic(1,1));
					pt.z = depth;
					objCloud->points.push_back(pt);
				}
		}
	}

	/********************************* function: convert3dUnOrganizedRGB ***********************************
	*******************************************************************************************************/

	void convert3dUnOrganizedRGB(cv::Mat &objDepth, cv::Mat &colorImage, Eigen::Matrix3f &camIntrinsic, PointCloudRGB::Ptr objCloud){
		int imgWidth = objDepth.cols;
		int imgHeight = objDepth.rows;
		
		for(int u=0; u<imgHeight; u++)
			for(int v=0; v<imgWidth; v++){
				float depth = objDepth.at<float>(u,v);
				cv::Vec3b colour = colorImage.at<cv::Vec3b>(u,v);
				if(depth > 0.1 && depth < 2.0){
					pcl::PointXYZRGB pt;
					pt.x = (float)((v - camIntrinsic(0,2)) * depth / camIntrinsic(0,0));
					pt.y = (float)((u - camIntrinsic(1,2)) * depth / camIntrinsic(1,1));
					pt.z = depth;
					uint32_t rgb = ((uint32_t)colour.val[2] << 16 | (uint32_t)colour.val[1] << 8 | (uint32_t)colour.val[0]);
					pt.rgb = *reinterpret_cast<float*>(&rgb);
					objCloud->points.push_back(pt);
				}
		}
	}

	/********************************* function: convert2d **************************************************
	*******************************************************************************************************/

	void convert2d(cv::Mat &objDepth, Eigen::Matrix3f &camIntrinsic, PointCloud::Ptr objCloud){
		for(int i=0;i<objCloud->points.size();i++){
			Eigen::Vector3f point2D = camIntrinsic * 
					Eigen::Vector3f(objCloud->points[i].x, objCloud->points[i].y, objCloud->points[i].z);
			point2D[0] = point2D[0]/point2D[2];
			point2D[1] = point2D[1]/point2D[2];
			if(point2D[1] > 0 && point2D[1] < objDepth.rows && point2D[0] > 0 && point2D[0] < objDepth.cols){
				if(objDepth.at<float>(point2D[1],point2D[0]) == 0  || point2D[2] < objDepth.at<float>(point2D[1],point2D[0]))
					objDepth.at<float>(point2D[1],point2D[0]) = (float)point2D[2];
			}
		}
	}

	/********************************* function: simpleVis **************************************************
	Reference: http://pointclouds.org/documentation/tutorials/pcl_visualizer.php
	*******************************************************************************************************/

	// 
	boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis (PointCloud::ConstPtr cloud){
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
		viewer->setBackgroundColor (0, 0, 0);
		viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
		viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
		viewer->initCameraParameters ();
		return (viewer);
	}

	/********************************* function: TransformPolyMesh *****************************************
	*******************************************************************************************************/

	void TransformPolyMesh(const pcl::PolygonMesh::Ptr &mesh_in, pcl::PolygonMesh::Ptr &mesh_out, Eigen::Matrix4f transform) {
		PointCloud::Ptr cloud_in (new PointCloud);
		PointCloud::Ptr cloud_out (new PointCloud);
		pcl::fromPCLPointCloud2(mesh_in->cloud, *cloud_in);
		pcl::transformPointCloud(*cloud_in, *cloud_out, transform);
		*mesh_out = *mesh_in;
		pcl::toPCLPointCloud2(*cloud_out, mesh_out->cloud);
		return;
	}

	/********************************* function: convertToMatrix *******************************************
	*******************************************************************************************************/

	void convertToMatrix(Eigen::Isometry3d &from, Eigen::Matrix4f &to){
		for(int ii=0; ii<4; ii++)
			for(int jj=0; jj<4; jj++)
				to(ii,jj) = from.matrix()(ii, jj);
	}

	/********************************* function: convertToIsometry3d ***************************************
	*******************************************************************************************************/

	void convertToIsometry3d(Eigen::Matrix4f &from, Eigen::Isometry3d &to){
		for(int ii=0; ii<4; ii++)
			for(int jj=0; jj<4; jj++)
				to.matrix()(ii,jj) = from(ii, jj);
	}

	/********************************* function: convertToWorld ********************************************
	*******************************************************************************************************/

	void convertToWorld(Eigen::Matrix4f &transform, Eigen::Matrix4f &cam_pose){
		transform = cam_pose*transform.eval();
	}

	/********************************* function: invertTransformationMatrix *******************************
	*******************************************************************************************************/
	void invertTransformationMatrix(Eigen::Matrix4f &tform){
		Eigen::Matrix3f rotm;
		Eigen::Vector3f trans;
		for(int ii = 0;ii < 3; ii++)
			for(int jj=0; jj < 3; jj++)
				rotm(ii,jj) = tform(ii,jj);
		trans[0] = tform(0,3);
		trans[1] = tform(1,3);
		trans[2] = tform(2,3);

		rotm = rotm.inverse().eval();
		trans = rotm*trans;

		for(int ii = 0;ii < 3; ii++)
			for(int jj=0; jj < 3; jj++)
				tform(ii,jj) = rotm(ii,jj);
		tform(0,3) = -trans[0];
		tform(1,3) = -trans[1];
		tform(2,3) = -trans[2];
	}

	/********************************* function: convertToCamera *******************************************
	*******************************************************************************************************/

	void convertToCamera(Eigen::Matrix4f &tform, Eigen::Matrix4f &cam_pose){
		Eigen::Matrix4f invCam(cam_pose);

		invertTransformationMatrix(invCam);
		tform = invCam*tform.eval();
	}

	/********************************* function: toEulerianAngle *******************************************
	from wikipedia
	*******************************************************************************************************/

	static void toEulerianAngle(Eigen::Quaternionf& q, Eigen::Vector3f& eulAngles){
		// roll (x-axis rotation)
		double sinr = +2.0 * (q.w() * q.x() + q.y() * q.z());
		double cosr = +1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y());
		eulAngles[0] = atan2(sinr, cosr);

		// pitch (y-axis rotation)
		double sinp = +2.0 * (q.w() * q.y() - q.z() * q.x());
		if (fabs(sinp) >= 1)
			eulAngles[1] = copysign(M_PI / 2, sinp); // use 90 degrees if out of range
		else
			eulAngles[1] = asin(sinp);

		// yaw (z-axis rotation)
		double siny = +2.0 * (q.w() * q.z() + q.x() * q.y());
		double cosy = +1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());  
		eulAngles[2] = atan2(siny, cosy);
	}
	
	/********************************* function: toQuaternion **********************************************
	from wikipedia
	*******************************************************************************************************/

	void toQuaternion(Eigen::Vector3f& eulAngles, Eigen::Quaternionf& q){
		double roll = eulAngles[0];
		double pitch = eulAngles[1];
		double yaw = eulAngles[2];
		double cy = cos(yaw * 0.5);
		double sy = sin(yaw * 0.5);
		double cr = cos(roll * 0.5);
		double sr = sin(roll * 0.5);
		double cp = cos(pitch * 0.5);
		double sp = sin(pitch * 0.5);

		q.w() = cy * cr * cp + sy * sr * sp;
		q.x() = cy * sr * cp - sy * cr * sp;
		q.y() = cy * cr * sp + sy * sr * cp;
		q.z() = sy * cr * cp - cy * sr * sp;
	}
		
	/********************************* function: toTransformationMatrix ************************************
	*******************************************************************************************************/

	void toTransformationMatrix(Eigen::Matrix4f& camPose, std::vector<double> camPose7D){
		camPose(0,3) = camPose7D[0];
		camPose(1,3) = camPose7D[1];
		camPose(2,3) = camPose7D[2];
		camPose(3,3) = 1;

		Eigen::Quaternionf q;
		q.w() = camPose7D[3];
		q.x() = camPose7D[4];
		q.y() = camPose7D[5];
		q.z() = camPose7D[6];
		Eigen::Matrix3f rotMat;
		rotMat = q.toRotationMatrix();

		for(int ii = 0;ii < 3; ii++)
			for(int jj=0; jj < 3; jj++){
				camPose(ii,jj) = rotMat(ii,jj);
			}
	}

	/********************************* function: rotationMatrixToEulerAngles *******************************
	*******************************************************************************************************/
	// Calculates rotation matrix to euler angles
	// The result is the same as MATLAB except the order
	// of the euler angles ( x and z are swapped ).
	Eigen::Vector3f rotationMatrixToEulerAngles(Eigen::Matrix3f R) {
	    float sy = sqrt(R(0,0) * R(0,0) +  R(1,0) * R(1,0) );
	    bool singular = sy < 1e-6;
	    float x, y, z;
	    if (!singular) {
	        x = atan2(R(2,1) , R(2,2));
	        y = atan2(-R(2,0), sy);
	        z = atan2(R(1,0), R(0,0));
	    }
	    else {
	        x = atan2(-R(1,2), R(1,1));
	        y = atan2(-R(2,0), sy);
	        z = 0;
	    }
	    Eigen::Vector3f rot;
	    rot << x, y, z;
	    return rot;
	}

	/********************************* function: getEMDError ***********************************************
	*******************************************************************************************************/

	void getEMDError(Eigen::Matrix4f testPose, Eigen::Matrix4f gtPose, PointCloud::Ptr objModel, float &error,
		std::pair<float, float> &xrange, std::pair<float, float> &yrange, std::pair<float, float> &zrange){

    	PointCloud::Ptr pcl_1 (new PointCloud);
    	PointCloud::Ptr pcl_2 (new PointCloud);
    	pcl::transformPointCloud(*objModel, *pcl_1, testPose);
    	pcl::transformPointCloud(*objModel, *pcl_2, gtPose);

    	int num_rows = pcl_1->points.size();
		cv::Mat xyzPts_1(num_rows, 1, CV_32FC3);
		cv::Mat xyzPts_2(num_rows, 1, CV_32FC3);

    	for(int ii=0; ii<num_rows; ii++){
			xyzPts_1.at<cv::Vec3f>(ii,0)[0] = pcl_1->points[ii].x;
		    xyzPts_1.at<cv::Vec3f>(ii,0)[1] = pcl_1->points[ii].y;
		    xyzPts_1.at<cv::Vec3f>(ii,0)[2] = pcl_1->points[ii].z;

		    xyzPts_2.at<cv::Vec3f>(ii,0)[0] = pcl_2->points[ii].x;
		    xyzPts_2.at<cv::Vec3f>(ii,0)[1] = pcl_2->points[ii].y;
		    xyzPts_2.at<cv::Vec3f>(ii,0)[2] = pcl_2->points[ii].z;

		}
		cv::MatND hist_1, hist_2;
		int xbins = numBinsEMD, ybins = numBinsEMD, zbins = numBinsEMD;
		int histSize[] = {xbins, ybins, zbins};
	  	float xranges[] = {xrange.first, xrange.second};
		float yranges[] = {yrange.first, yrange.second};
		float zranges[] = {zrange.first, zrange.second};
		int channels[] = {0, 1, 2};
		const float* ranges[] = { xranges, yranges, zranges};
		
	    cv::calcHist( &xyzPts_1, 1, channels, cv::Mat(), hist_1, 3, histSize, ranges, true, false);
	    cv::calcHist( &xyzPts_2, 1, channels, cv::Mat(), hist_2, 3, histSize, ranges, true, false);

	    // std::cout << xrange.first << " " << yrange.first << " " << zrange.first << " " << xrange.second << " " << yrange.second << " " << zrange.second << std::endl;
	    int sigSize = xbins*ybins*zbins;
	  	cv::Mat sig1(sigSize, 4, CV_32FC1);
	  	cv::Mat sig2(sigSize, 4, CV_32FC1);

		//fill value into signature
		for(int x=0; x<xbins; x++) {
			for(int y=0; y<ybins; y++) {
		    	for(int z=0; z<zbins; z++) {
			        float binval = hist_1.at<float>(x,y,z);
			        sig1.at<float>( x*ybins*zbins + y*zbins + z, 0) = binval;
			        sig1.at<float>( x*ybins*zbins + y*zbins + z, 1) = x;
			        sig1.at<float>( x*ybins*zbins + y*zbins + z, 2) = y;
			        sig1.at<float>( x*ybins*zbins + y*zbins + z, 3) = z;

			        binval = hist_2.at<float>(x,y,z);
			        sig2.at<float>( x*ybins*zbins + y*zbins + z, 0) = binval;
			        sig2.at<float>( x*ybins*zbins + y*zbins + z, 1) = x;
			        sig2.at<float>( x*ybins*zbins + y*zbins + z, 2) = y;
			        sig2.at<float>( x*ybins*zbins + y*zbins + z, 3) = z;
		        }
		    }
		}

	   error = cv::EMD(sig1, sig2, CV_DIST_L2); //emd 0 is best matching.   
	}

	/********************************* function: c_dist_pose **********************************************
	*******************************************************************************************************/

	// float c_dist_pose(Eigen::Matrix4f pose_1, Eigen::Matrix4f pose_2, PointCloud::Ptr objModel) {
	//   size_t number_of_points = objModel->points.size();

	//   float max_distance = 0;
	//   for(int ii=0; ii<number_of_points; ii++){
	//     float min_distance = FLT_MAX;

	//     Eigen::Matrix<Scalar, 3, 1> p = (allTransforms[index_1]*hull_Q_3D[ii].pos().homogeneous()).head<3>();
	//     for(int jj=0; jj<number_of_points; jj++){
	//       Eigen::Matrix<Scalar, 3, 1> q = (allTransforms[index_2]*hull_Q_3D[jj].pos().homogeneous()).head<3>();
	//       float dist = (p - q).norm();
	//       if(dist < min_distance)
	//         min_distance = dist;
	//     }
	    
	//     if(min_distance > max_distance)
	//       max_distance = min_distance;
	//   }

	//   return max_distance;
	// }

	/********************************* function: getPoseError **********************************************
	*******************************************************************************************************/

	void getPoseError(Eigen::Matrix4f testPose, Eigen::Matrix4f gtPose, Eigen::Vector3f symInfo, 
		float &meanrotErr, float &transErr){
		Eigen::Matrix3f testRot, gtRot, rotdiff;
		for(int ii = 0;ii < 3; ii++)
			for(int jj=0; jj < 3; jj++){
				testRot(ii,jj) = testPose(ii,jj);
				gtRot(ii,jj) = gtPose(ii,jj);
			}

		testRot = testRot.inverse().eval();
		rotdiff = testRot*gtRot;
		Eigen::Quaternionf rotdiffQ(rotdiff);
		Eigen::Vector3f rotErrXYZ;
		toEulerianAngle(rotdiffQ, rotErrXYZ);
		rotErrXYZ = rotErrXYZ*180.0/M_PI;

		for(int dim = 0; dim < 3; dim++){
			rotErrXYZ(dim) = fabs(rotErrXYZ(dim));
			if (symInfo(dim) == 90){
				rotErrXYZ(dim) = abs(rotErrXYZ(dim) - 90);
				rotErrXYZ(dim) = std::min(rotErrXYZ(dim), 90 - rotErrXYZ(dim));
			}
			else if(symInfo(dim) == 180){
				rotErrXYZ(dim) = std::min(rotErrXYZ(dim), 180 - rotErrXYZ(dim));
			}
			else if(symInfo(dim) == 360){
				rotErrXYZ(dim) = 0;
			}
		}

		meanrotErr = (rotErrXYZ(0) + rotErrXYZ(1) + rotErrXYZ(2))/3;
		transErr = sqrt(pow(gtPose(0,3) - testPose(0,3), 2) + 
			pow(gtPose(1,3) - testPose(1,3), 2) + 
				pow(gtPose(2,3) - testPose(2,3), 2));
	}

	/******************************** function: getRotDistance *********************************************
	*******************************************************************************************************/

	float getRotDistance(Eigen::Matrix3f rotMat1, Eigen::Matrix3f rotMat2, Eigen::Vector3f symInfo){
		Eigen::Matrix3f rotdiff;

		rotdiff = rotMat1*rotMat2;
		Eigen::Vector3f rotErrXYZ;
		rotErrXYZ = rotationMatrixToEulerAngles(rotdiff);
		rotErrXYZ = rotErrXYZ*180.0/M_PI;

		for(int dim = 0; dim < 3; dim++){
			rotErrXYZ(dim) = fabs(rotErrXYZ(dim));
			if (symInfo(dim) == 90){
				rotErrXYZ(dim) = abs(rotErrXYZ(dim) - 90);
				rotErrXYZ(dim) = std::min(rotErrXYZ(dim), 90 - rotErrXYZ(dim));
			}
			else if(symInfo(dim) == 180){
				rotErrXYZ(dim) = std::min(rotErrXYZ(dim), 180 - rotErrXYZ(dim));
			}
			else if(symInfo(dim) == 360){
				rotErrXYZ(dim) = 0;
			}
		}

		float meanrotErr = (rotErrXYZ(0) + rotErrXYZ(1) + rotErrXYZ(2))/3;
		return meanrotErr;
	}

	/********************************* function: convertToCVMat ********************************************
	*******************************************************************************************************/

	void convertToCVMat(Eigen::Matrix4f &pose, cv::Mat &cvPose){
		cvPose.at<float>(0, 0) = pose(0,3);
		cvPose.at<float>(0, 1) = pose(1,3);
		cvPose.at<float>(0, 2) = pose(2,3);

		Eigen::Matrix3f rotMat;
		for(int ii = 0;ii < 3; ii++)
			for(int jj=0; jj < 3; jj++){
				rotMat(ii,jj) = pose(ii,jj);
			}
		Eigen::Quaternionf rotMatQ(rotMat);
		Eigen::Vector3f rotErrXYZ;
		toEulerianAngle(rotMatQ, rotErrXYZ);
		rotErrXYZ = rotErrXYZ*180.0/M_PI;

		cvPose.at<float>(0, 3) = rotErrXYZ(0);
		cvPose.at<float>(0, 4) = rotErrXYZ(1);
		cvPose.at<float>(0, 5) = rotErrXYZ(2);
	}

	/********************************* function: convert6DToMatrix *****************************************
	*******************************************************************************************************/

	void convert6DToMatrix(Eigen::Matrix4f &pose, cv::Mat &points, int index){
		pose.setIdentity();
		pose(0,3) = points.at<float>(index, 0);
		pose(1,3) = points.at<float>(index, 1);
		pose(2,3) = points.at<float>(index, 2);

		Eigen::Matrix3f rotMat;
		Eigen::Quaternionf q;
		Eigen::Vector3f rotXYZ;
		rotXYZ << points.at<float>(index, 3) * M_PI/180.0, 
					points.at<float>(index, 4) * M_PI/180.0,
					points.at<float>(index, 5) * M_PI/180.0;
		toQuaternion(rotXYZ, q);
		rotMat = q.toRotationMatrix();

      	for(int ii = 0;ii < 3; ii++)
			for(int jj=0; jj < 3; jj++){
				pose(ii,jj) = rotMat(ii,jj);
			}
	}

	/********************************* function: writePoseToFile *******************************************
	*******************************************************************************************************/
	
	void writePoseToFile(Eigen::Matrix4f pose, std::string objName, std::string scenePath, std::string filename){
		ofstream pFile;
		pFile.open ((scenePath +  filename + "_" + objName + ".txt").c_str(), std::ofstream::out | std::ofstream::app);
		pFile << pose(0,0) << " " << pose(0,1) << " " << pose(0,2) << " " << pose(0,3) 
					 << " " << pose(1,0) << " " << pose(1,1) << " " << pose(1,2) << " " << pose(1,3)
					 << " " << pose(2,0) << " " << pose(2,1) << " " << pose(2,2) << " " << pose(2,3) << std::endl;
		pFile.close();
	}

	/********************************* function: writeScoreToFile ******************************************
	*******************************************************************************************************/

	void writeScoreToFile(float score, std::string objName, std::string scenePath, std::string filename){
		ofstream pFile;
		pFile.open ((scenePath +  filename + "_" + objName + ".txt").c_str(), std::ofstream::out | std::ofstream::app);
		pFile << score << std::endl;
		pFile.close();
	}

	/********************************* function: performTrICP **********************************************
	*******************************************************************************************************/

	void performTrICP(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclSegment, 
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclModel, 
		Eigen::Isometry3d &currTransform,
		Eigen::Isometry3d &finalTransform,
		float trimPercentage){
		PointCloud::Ptr modelCloud (new PointCloud);
		PointCloud::Ptr segmentCloud (new PointCloud);

		Eigen::Matrix4f tform;
		
		copyPointCloud(*pclModel, *modelCloud);
		copyPointCloud(*pclSegment, *segmentCloud);

		// initialize trimmed ICP
		pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
		pcl::recognition::TrimmedICP<pcl::PointXYZ, float> tricp;
		tricp.init(modelCloud);
		tricp.setNewToOldEnergyRatio(1.f);

		float numPoints = trimPercentage*segmentCloud->points.size();
		
		// get current object transform
		utilities::convertToMatrix(currTransform, tform);
		tform = tform.inverse().eval();

		tricp.align(*segmentCloud, abs(numPoints), tform);
		tform = tform.inverse().eval();

		utilities::convertToIsometry3d(tform, finalTransform);
	}

	/********************************* function: pointToPlaneICP *******************************************
	*******************************************************************************************************/

	void pointToPlaneICP(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclSegment, 
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclModel, 
		Eigen::Matrix4f &offsetTransform){

		PointCloudRGBNormal::Ptr modelCloud (new PointCloudRGBNormal);
		PointCloudRGBNormal::Ptr segmentCloud (new PointCloudRGBNormal);
		PointCloudRGBNormal segCloudTrans;
		copyPointCloud(*pclModel, *modelCloud);
		copyPointCloud(*pclSegment, *segmentCloud);

	    pcl::IterativeClosestPointWithNormals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>::Ptr icp ( new pcl::IterativeClosestPointWithNormals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> () );
   	    icp->setMaximumIterations ( 100 );
   	    icp->setInputSource ( segmentCloud ); // not cloud_source, but cloud_source_trans!
   	    icp->setInputTarget ( modelCloud );
   	    icp->align ( segCloudTrans );
   	    if ( icp->hasConverged() ) {
   	  	  offsetTransform = icp->getFinalTransformation();
     	  std::cout << "ICP score: " << icp->getFitnessScore() << std::endl;
   		}
   	    else {
     		std::cout << "ICP did not converge." << std::endl;
     		offsetTransform << 1, 0, 0, 0,
     						   0, 1, 0, 0,
     						   0, 0, 1, 0,
     						   0, 0, 0, 1;
   	    }
     }

     /********************************* function: pointMatcherICP *******************************************
	*******************************************************************************************************/

	void pointMatcherICP(std::string refPath, 
		std::string dataPath, 
		Eigen::Matrix4f &offsetTransform){

		const DP ref(DP::load(refPath));
		const DP data(DP::load(dataPath));

		PM::ICP icp;
		icp.setDefault();
		PM::TransformationParameters T = icp(data, ref);
		std::cout << "Final transformation:" << endl << T << endl;

		offsetTransform << T(0, 0), T(0, 1), T(0, 2), T(0, 3), 
						   T(1, 0), T(1, 1), T(1, 2), T(1, 3), 
						   T(2, 0), T(2, 1), T(2, 2), T(2, 3), 
						   0, 0, 0, 1;
     }
	/********************************* end of functions ****************************************************
	*******************************************************************************************************/

} // namespace