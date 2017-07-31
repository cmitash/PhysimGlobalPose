#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <boost/shared_ptr.hpp>
#include <camera_constants.h>
#include <simulation_io.hpp>

// For OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace Eigen;
using namespace pcl;
using namespace pcl::console;
using namespace pcl::io;
using namespace pcl::simulation;
using namespace std;

SimExample::Ptr simexample;

void TransformPolyMesh(const pcl::PolygonMesh::Ptr &mesh_in, pcl::PolygonMesh::Ptr &mesh_out, Eigen::Matrix4f transform) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new
                                                pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new
                                                 pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromPCLPointCloud2(mesh_in->cloud, *cloud_in);

  transformPointCloud(*cloud_in, *cloud_out, transform);

  *mesh_out = *mesh_in;
  pcl::toPCLPointCloud2(*cloud_out, mesh_out->cloud);
  return;
}

int
main (int argc, char** argv)
{
  print_info ("Render depth image");
  std::vector<unsigned short> depth_image;

  int width = kCameraWidth;
  int height = kCameraHeight;
  simexample = SimExample::Ptr (new SimExample (argc, argv, height,width));
// 
  pcl::simulation::Scene::Ptr scene_;
  scene_ = simexample->scene_;
  if (scene_ == NULL) {
    printf("ERROR: Scene is not set\n");
  }

  scene_->clear();

  pcl::PolygonMesh mesh;
  pcl::io::loadPolygonFile ("crayola_24_ct.obj", mesh);
  pcl::PolygonMesh::Ptr mesh_in (new pcl::PolygonMesh (mesh));
  pcl::PolygonMesh::Ptr mesh_out (new pcl::PolygonMesh (mesh));
  Eigen::Matrix4f transform;
  transform << 1, 0, 0, 0.78,
               0, 1, 0, 0,
               0, 0, 1, 0.55;
               0, 0, 0, 1;
  TransformPolyMesh(mesh_in, mesh_out, transform);
  PolygonMeshModel::Ptr transformed_mesh = PolygonMeshModel::Ptr (new PolygonMeshModel (GL_POLYGON, mesh_out));
  scene_->add (transformed_mesh);

  pcl::PolygonMesh mesh1;  // (new pcl::PolygonMesh);
  pcl::io::loadPolygonFile ("kleenex_tissue_box.obj", mesh1);
  pcl::PolygonMesh::Ptr mesh_in1 (new pcl::PolygonMesh (mesh1));
  pcl::PolygonMesh::Ptr mesh_out1 (new pcl::PolygonMesh (mesh1));
  Eigen::Matrix4f transform1;
  transform1 << 1, 0, 0, 0.88,
               0, 1, 0, -0.05,
               0, 0, 1, 0.55;
               0, 0, 0, 1;
  TransformPolyMesh(mesh_in1, mesh_out1, transform1);
  PolygonMeshModel::Ptr transformed_mesh1 = PolygonMeshModel::Ptr (new PolygonMeshModel (GL_POLYGON, mesh_out1));
  scene_->add (transformed_mesh1);

  Eigen::Isometry3d camera_pose;
  camera_pose.setIdentity();

  

  Eigen::Vector3d trans;
  Eigen::Matrix3d rot;

  rot << -3.39650251e-02,  -7.07316995e-01,    7.06080079e-01,
         -9.81231213e-01,   1.57783359e-01,    1.10858969e-01,
         -1.89820126e-01,  -6.89062476e-01,   -6.99400604e-01;
  camera_pose = camera_pose*rot;

  Matrix3d m;
  m = AngleAxisd(0, Vector3d::UnitZ())     * AngleAxisd(0, Vector3d::UnitY())    * AngleAxisd(M_PI/2, Vector3d::UnitX()); 
  camera_pose *= m;
  m = AngleAxisd(M_PI/2, Vector3d::UnitZ())     * AngleAxisd(0, Vector3d::UnitY())    * AngleAxisd(0, Vector3d::UnitX()); 
  camera_pose *= m;

  trans << 5.47767639e-01, -1.38075694e-01, 8.96193922e-01;
  camera_pose.translation() = trans;

  std::cout<<"camera pose: "<<camera_pose.matrix()<<std::endl;
  simexample->doSim(camera_pose);
  const float *depth_buffer = simexample->rl_->getDepthBuffer();
  simexample->get_depth_image_uint(depth_buffer, &depth_image);
  simexample->write_depth_image(depth_buffer, "abc.png");

  // for (size_t ii = 0; ii < 307200; ++ii){
  //   std::cout<<depth_image.at(ii)<<std::endl;
  // }
  return 0;
}
