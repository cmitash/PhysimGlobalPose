#include <PoseVisualization.hpp>

namespace pose_visualization{

	PoseVisualization::PoseVisualization(){
		viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("3D Viewer"));
		viewer->setBackgroundColor (0, 0, 0);
	}

	PoseVisualization::~PoseVisualization(){}

	void PoseVisualization::startViz() {

		while (!viewer->wasStopped ()) {
	    	viewer->spinOnce (100);
	    	boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  		}

		viewer->close(); 
	}

	void PoseVisualization::loadSceneCloud(cv::Mat depthImage, cv::Mat colorImage, Eigen::Matrix3f camIntrinsic, Eigen::Matrix4f camPose){
		PointCloudRGB::Ptr sceneCloud = PointCloudRGB::Ptr(new PointCloudRGB);
		utilities::convert3dOrganizedRGB(depthImage, colorImage, camIntrinsic, sceneCloud);
		pcl::transformPointCloud(*sceneCloud, *sceneCloud, camPose);
		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(sceneCloud);
		viewer->addPointCloud<pcl::PointXYZRGB> (sceneCloud, rgb, "sample cloud");
		viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
		viewer->initCameraParameters ();
	}
	
	void PoseVisualization::loadObjectModels(pcl::PolygonMesh meshObj, std::vector<double> poseObj, std::string meshName){
		Eigen::Matrix4f objTransform = Eigen::Matrix4f::Zero(4,4);
		pcl::PolygonMesh::Ptr mesh_in (new pcl::PolygonMesh (meshObj));
  		pcl::PolygonMesh::Ptr mesh_out (new pcl::PolygonMesh (meshObj));
		utilities::toTransformationMatrix(objTransform, poseObj);
		utilities::TransformPolyMesh(mesh_in, mesh_out, objTransform);
		viewer->addPolygonMesh(*mesh_out, meshName);
		viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, (double)rand()/(double)RAND_MAX,(double)rand()/(double)RAND_MAX,(double)rand()/(double)RAND_MAX, meshName);
		viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, 0.4, meshName);
		viewer->resetCameraViewpoint("sample cloud");
	}

	void PoseVisualization::setCamera() {
		viewer->setCameraPosition(-0.634251, 0.232444, 0.124079, 0.487722, 0.0377538, -0.0502087, 0.14127, -0.0698937, 0.987501); 
		viewer->setCameraFieldOfView(49.1311*M_PI/180); 
		viewer->setCameraClipDistances(0.399983, 2.39047); 
		viewer->setPosition(1985, 24); 
		viewer->setSize(1855, 1056);
		viewer->updateCamera();
	}
}