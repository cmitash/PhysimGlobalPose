#include <ros/ros.h>
#include <detection_package/UpdateActiveListFrame.h>
#include <detection_package/UpdateBbox.h>

#include <cstdlib>
#include <vector>

#include <string>
#include <sstream>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/io/pcd_io.h>

#include <pcl/common/common_headers.h>
#include <pcl/conversions.h>
#include <pcl/point_types_conversion.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/io/obj_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>

#include <pcl/filters/crop_box.h>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudRGB;

double fx_d = 0.0;
double fy_d = 0.0;
double cx_d = 0.0;
double cy_d = 0.0;

std::string cam_mode = "hd";
std::string data_path = "/home/pracsys/github/get_images/";

enum apc_objects {
  crayola_24_ct = 1,
  expo_dry_erase_board_eraser = 2,
  folgers_classic_roast_coffee = 3,
  scotch_duct_tape = 4,
  dasani_water_bottle = 5,
  jane_eyre_dvd = 6,
  up_glucose_bottle = 7,
  laugh_out_loud_joke_book = 8,
  soft_white_lightbulb = 9,
  kleenex_tissue_box = 10,
  ticonderoga_12_pencils = 11,
  dove_beauty_bar = 12,
  dr_browns_bottle_brush = 13,
  elmers_washable_no_run_school_glue = 14,
  rawlings_baseball = 15
};

std::string apc_objects_strs[] = {
 "crayola_24_ct", "expo_dry_erase_board_eraser", "folgers_classic_roast_coffee",
  "scotch_duct_tape", "dasani_water_bottle", "jane_eyre_dvd",
  "up_glucose_bottle", "laugh_out_loud_joke_book", "soft_white_lightbulb",
  "kleenex_tissue_box", "ticonderoga_12_pencils", "dove_beauty_bar",
  "dr_browns_bottle_brush", "elmers_washable_no_run_school_glue", "rawlings_baseball"
};

boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> rgbVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->initCameraParameters ();
  return (viewer);
}

void storePointCloud(std::string filename, cv::Mat depthImage, cv::Rect bbox/*, PointCloudRGB::Ptr cloud_xyzrgb*/)
{
  std::string env_p;
  if(const char* env = std::getenv("PHYSIM_GLOBAL_POSE"))
  {
      std::cout << "Your PHYSIM_GLOBAL_POSE is: " << env << '\n';
      env_p = std::string(env);
  }
  else
  {
    std::cout<<"Please set PHYSIM_GLOBAL_POSE in bashrc"<<endl;
    exit(-1);
  }

  depthImage.convertTo(depthImage, CV_32F);
  PointCloud::Ptr in_cloud (new PointCloud);
  PointCloud::Ptr cloud_filtered (new PointCloud);

  std::cout<<"Camera Info before calculation :: fx: "<<fx_d<<" fy: "<<fy_d<<" cx: "<<cx_d<<" cy: "<<cy_d<<std::endl;
  for(int i=bbox.y; i<(bbox.y + bbox.height); i++)
    for(int j=bbox.x; j<(bbox.x + bbox.width); j++){
      double depth = depthImage.at<float>(i,j)/1000;
      if(depth  != 0){
        std::cout<<"depth :: "<<depth<<" "<<endl;
        Eigen::Vector3d point;
        point << ((j - cx_d) * depth * fx_d), 
            ((i - cy_d) * depth * fy_d), depth;
        pcl::PointXYZ pt(point[0], point[1], point[2]);
        in_cloud->points.push_back(pt);
    }
  }

  //Filter off points which are beyond 1m from camera
  Eigen::Vector4f minPoint(-0.32,-0.14,0.67 ,0);
  Eigen::Vector4f maxPoint(-0.008,0.06,1.11,0);
  pcl::CropBox<pcl::PointXYZ> cropFilterXYZ; 
  cropFilterXYZ.setInputCloud(in_cloud);
  cropFilterXYZ.setMin(minPoint);
  cropFilterXYZ.setMax(maxPoint);
  cropFilterXYZ.filter (*cloud_filtered);

  pcl::io::savePLYFileASCII(filename + "_pcl.ply", *cloud_filtered);

  /* Debug :: color point cloud */

  // PointCloudRGB::Ptr cloud_xyzrgb_out (new PointCloudRGB);
  // pcl::CropBox<pcl::PointXYZRGB> cropFilterXYZRGB; 
  // cropFilterXYZRGB.setInputCloud(cloud_xyzrgb);
  // cropFilterXYZRGB.setMin(minPoint);
  // cropFilterXYZRGB.setMax(maxPoint);
  // cropFilterXYZRGB.filter (*cloud_xyzrgb_out);

  //boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
  //viewer = rgbVis(cloud_xyzrgb_out);
  //viewer = simpleVis(cloud_filtered);

  // while (!viewer->wasStopped ())
  // {
  //   viewer->spinOnce (100);
  //   boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  // }
  /* DEBUG ENDS */

  // Generate Point Cloud for the object model
  pcl::PointCloud<pcl::PointXYZ>::Ptr orig_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PCLPointCloud2 pcl_orig_cloud;
  if (pcl::io::loadPCDFile<pcl::PointXYZ> (env_p + "/src/super4pcs/objmodels/" + filename + ".pcd", *orig_cloud) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
    exit(1);
  }
  toPCLPointCloud2(*orig_cloud, pcl_orig_cloud);
  pcl::io::savePLYFile(filename + "_orig.ply", pcl_orig_cloud);

  // Run Super4PCS
  std::string command = env_p + "/src/super4pcs/build/Super4PCS -i " + 
                        env_p + "/devel/lib/rcnn-detect/" + filename + "_orig.ply " + 
                        env_p + "/devel/lib/rcnn-detect/" + filename + "_pcl.ply -o 0.7 -d 0.05 -t 1000 -n 400 -r " + 
                        env_p + "/src/super4pcs/super4pcs_fast.obj -m " + env_p + "/src/super4pcs/mat_super4pcs_fast.txt";

  std::cout<<"Command : "<<command<<std::endl;
  system(command.c_str());
}



int main(int argc, char **argv)
{
  std::vector<apc_objects> active_object_list;

  ros::init(argc, argv, "test_detection");

  if(argc == 1){
    std::cout<<"Enter object to detect"<<std::endl;
    exit(1);
  }

  std::string to_detect(argv[1]);
  apc_objects index = (apc_objects)1;
  for(int i = 0;i<15;i++)
  {
    if(apc_objects_strs[i].compare(to_detect) == 0)
    {
      index = (apc_objects)(i+1);
      break;
    }
  }
  active_object_list.push_back(index);
  std::cout<<"Selected object is : "<<index<<std::endl;

  ros::NodeHandle n;
  ros::ServiceClient clientlist = n.serviceClient<detection_package::UpdateActiveListFrame>("/update_active_list");
  ros::ServiceClient clientbox = n.serviceClient<detection_package::UpdateBbox>("/update_bbox");
  detection_package::UpdateActiveListFrame listsrv;
  detection_package::UpdateBbox boxsrv;
  
  ros::Publisher detectPub = n.advertise<sensor_msgs::Image>("detectpub", 1000);
  
  // sensor_msgs::CameraInfo::ConstPtr cam_info_msg;
  // cam_info_msg = ros::topic::waitForMessage<sensor_msgs::CameraInfo>("/kinect2_right/" + cam_mode + "/camera_info", n);
  
  // fx_d = 1/cam_info_msg->K[0];
  // cx_d = cam_info_msg->K[2];
  // fy_d = 1/cam_info_msg->K[4];
  // cy_d = cam_info_msg->K[5];

  // std::cout<<"Camera Info :: fx: "<<fx_d<<" fy: "<<fy_d<<" cx: "<<cx_d<<" cy: "<<cy_d<<std::endl;
  for(int i=0;i<active_object_list.size();i++)
  {
    listsrv.request.active_list.push_back(active_object_list[i]);
  }
  
  if (clientlist.call(listsrv))
  {
    ROS_INFO("Returned : %d", (bool)listsrv.response.result);
  }
  else
  {
    ROS_ERROR("Failed to call service UpdateActiveListFrame");
    return 1;
  } 

  std::cout<<"Size of active objects : "<<active_object_list.size()<<std::endl;

  mkdir((data_path + to_detect).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

  std::ifstream filein((data_path + "scene_labels/" + to_detect + "_ground_truth.txt").c_str());
  std::ofstream fileout((data_path + to_detect + "/" + to_detect + "_result.txt").c_str(), std::ofstream::out | std::ofstream::app);
  std::string line;

  while (std::getline(filein, line))
  {
    std::stringstream ss(line);
    std::string buf;
    std::vector<std::string> tokens;
    while (ss >> buf)
        tokens.push_back(buf);

    std::string s_index(tokens[0]);
    std::cout<<"Detecting Image : "<<s_index<<endl;
    
    // sensor_msgs::Image::ConstPtr msg_color;
    // sensor_msgs::Image::ConstPtr msg_depth;

    // msg_color = ros::topic::waitForMessage<sensor_msgs::Image>("/kinect2_right/" + cam_mode + "/image_color_rect", n);
    // msg_depth = ros::topic::waitForMessage<  sensor_msgs::Image>("/kinect2_right/" + cam_mode + "/image_depth_rect", n);
    
    
    // cv_bridge::CvImagePtr cv_ptr_color;
    // cv_bridge::CvImagePtr cv_ptr_depth;

    // PointCloudRGB::Ptr cloud_xyzrgb(new PointCloudRGB);
    // try
    // {
    //    cv_ptr_color = cv_bridge::toCvCopy(*msg_color, sensor_msgs::image_encodings::BGR8);
    //    cv_ptr_depth = cv_bridge::toCvCopy(*msg_depth, (*msg_depth).encoding);
    //    sensor_msgs::PointCloud2::ConstPtr msg3 = 
    //    ros::topic::waitForMessage<sensor_msgs::PointCloud2>("/kinect2_right/" + cam_mode +"/points", n);
    //    pcl::PCLPointCloud2 pcl_pc2;
    //    pcl_conversions::toPCL(*msg3,pcl_pc2);
    //    pcl::fromPCLPointCloud2(pcl_pc2,*cloud_xyzrgb);    
    // }
    // catch (cv_bridge::Exception& e)
    // {
    //     ROS_ERROR("cv_bridge exception: %s", e.what());\
    //     exit(-1);
    // }
    cv::Mat color_image, depth_image;
    color_image = cv::imread(data_path + "scene_images/color_image" + s_index + ".png", CV_LOAD_IMAGE_COLOR);
    depth_image = cv::imread(data_path + "scene_images/depth_image" + s_index + ".png", CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);

    cv::imwrite(data_path + "testing.png",color_image);
    if (clientbox.call(boxsrv))
    {
      for(int i=0;i<active_object_list.size();i++){
        cv::Rect bbox(boxsrv.response.tl_x[i],
                      boxsrv.response.tl_y[i],
                      boxsrv.response.br_x[i] - boxsrv.response.tl_x[i],
                      boxsrv.response.br_y[i] - boxsrv.response.tl_y[i]);

         std::string text = apc_objects_strs[active_object_list[i] - 1];
         int fontFace = 1;
         double fontScale = 1;
         int thickness = 2;  
         int red = rand() % 256;
         int green = rand() % 256;
         int blue = rand() % 256;
         cv::Point textOrg(boxsrv.response.tl_x[i], boxsrv.response.tl_y[i]);

         // cv::putText(cv_ptr_color->image, text, textOrg, fontFace, fontScale, cv::Scalar(red, green, blue), thickness,8);
         // std::cout<<"Bounding Box is : "<<bbox<<std::endl;
         // cv::rectangle(cv_ptr_color->image, bbox, cv::Scalar(red, green, blue), 2);
         // storePointCloud(text, cv_ptr_depth->image, bbox, cloud_xyzrgb);

         cv::putText(color_image, text, textOrg, fontFace, fontScale, cv::Scalar(red, green, blue), thickness,8);
         std::cout<<"Bounding Box is : "<<bbox<<std::endl;
         cv::rectangle(color_image, bbox, cv::Scalar(red, green, blue), 2);
         // storePointCloud(text, depth_image, bbox);
         fileout << atoi(s_index.c_str()) << " " <<
                    bbox.tl().x << " " << 
                    bbox.tl().y << " " << 
                    bbox.br().x << " " << 
                    bbox.br().y << " " <<
                    boxsrv.response.scores[i]<<                   
                    std::endl;
      }
      // detectPub.publish(cv_ptr_color->toImageMsg());
      // detectPub.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", color_image).toImageMsg());
      cv::imwrite(data_path + to_detect + "/result" + s_index + ".png", color_image);
    }
    else
    {
      ROS_ERROR("Failed to call service UpdateBbox");
      return 1;
    }
  }
  
  filein.close();
  fileout.close();

  std::string cp1 = "cp /media/pracsys/DATA/ICRA2017_Testing/scene_test/scene_labels/" + to_detect + "*.txt " + data_path + to_detect + "/";
  std::cout<<"Command : "<<cp1<<std::endl;
  system(cp1.c_str());

  std::string cp2 = "cp " + data_path + "compute_accuracy.py " + data_path + to_detect + "/";
  std::cout<<"Command : "<<cp2<<std::endl;
  system(cp2.c_str());

  // std::string cd = "cd " + data_path + to_detect;
  // std::cout<<"Command : "<<cd<<std::endl;
  // system(cd.c_str());

  // std::string exe = "python compute_accuracy.py " + to_detect + "_ground_truth.txt " + to_detect + "_result.txt";
  // std::cout<<"Command : "<<exe<<std::endl;
  // system(exe.c_str());

  return 0;
}