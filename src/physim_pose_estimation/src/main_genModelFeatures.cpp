#include <common_io.h>

int main(int argc, char* argv[]) {

	if(argc < 3){
		std::cout << "<model_location> <voxel_size>" << std::endl;
	}

	std::map<std::vector<float>, float> PPFMap;
	float max_count_ppf;
	
	float transDisc = 0.01;
	float rotDisc = 10;
	float number_of_samples = 300;

	float samplingDisc = atof(argv[2]);

	pcl::PointCloud<pcl::PointXYZ>::Ptr model_cloud (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_normals_model (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_sampled (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

	pcl::io::loadPCDFile((std::string(argv[1]) + "/textured.pcd").c_str(), *model_cloud);

	clock_t t0 = clock();
	std::cout << "read point cloud. size: " << model_cloud->points.size() << std::endl;

	copyPointCloud(*model_cloud, *cloud_normals_model);

	pcl::NormalEstimation<pcl::PointXYZ, pcl::PointXYZRGBNormal> ne;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
	ne.setInputCloud (model_cloud);
	ne.setSearchMethod (tree);
	ne.setRadiusSearch (0.005);
	ne.compute (*cloud_normals_model);

	std::cout << "Time after normal estimation: " << float( clock () - t0 ) /  CLOCKS_PER_SEC << std::endl;

	pcl::VoxelGrid<pcl::PointXYZRGBNormal> sor;
	sor.setInputCloud (cloud_normals_model);
	sor.setLeafSize (samplingDisc, samplingDisc, samplingDisc);
	sor.filter (*cloud_normals_model);

    int sample_fraction_Q = std::max(1, static_cast<int>(cloud_normals_model->points.size() / number_of_samples));

    for (int i = 0; i < cloud_normals_model->points.size(); ++i) {
        if (rand() % sample_fraction_Q == 0) {
            cloud_sampled->points.push_back(cloud_normals_model->points[i]);
        }
    }

	std::cout << "point after sampling: " << cloud_sampled->points.size() << std::endl;

	float max_val = 1;
	for(int ii=0; ii< cloud_sampled->points.size(); ii++)
		for(int jj=0; jj< cloud_sampled->points.size(); jj++){
			Eigen::Vector3f p_1(cloud_sampled->points[ii].x, cloud_sampled->points[ii].y,
				cloud_sampled->points[ii].z);
			Eigen::Vector3f p_2(cloud_sampled->points[jj].x, cloud_sampled->points[jj].y,
				cloud_sampled->points[jj].z);
			Eigen::Vector3f d = p_1 - p_2;
			Eigen::Vector3f n_1(cloud_sampled->points[ii].normal_x, 
				cloud_sampled->points[ii].normal_y, cloud_sampled->points[ii].normal_z);
			Eigen::Vector3f n_2(cloud_sampled->points[jj].normal_x, 
				cloud_sampled->points[jj].normal_y, cloud_sampled->points[jj].normal_z);

			float ppf_1 = d.lpNorm<2>();
			float ppf_2 = atan2(n_1.cross(d).lpNorm<2>(), n_1.dot(d))*180/M_PI;
			float ppf_3 = atan2(n_2.cross(d).lpNorm<2>(), n_2.dot(d))*180/M_PI;
			float ppf_4 = atan2(n_1.cross(n_2).lpNorm<2>(), n_1.dot(n_2))*180/M_PI;

			ppf_1 = ppf_1 - fmod(ppf_1, transDisc);
			ppf_2 = ppf_2 - fmod(ppf_2, rotDisc);
			ppf_3 = ppf_3 - fmod(ppf_3, rotDisc);
			ppf_4 = ppf_4 - fmod(ppf_4, rotDisc);

			if(isnan(ppf_1) || isnan(ppf_2) || isnan(ppf_3) || isnan(ppf_4)){
				// std::cout << "p1: " << p_1 << std::endl;
				// std::cout << "p2: " << p_2 << std::endl;
				// std::cout << "n1: " << n_1 << std::endl;
				// std::cout << "n2: " << n_2 << std::endl;
			}

			std::vector<float> ppf_;
			ppf_.push_back(ppf_1);
			ppf_.push_back(ppf_2);
			ppf_.push_back(ppf_3);
			ppf_.push_back(ppf_4);

			std::map<std::vector<float>, float>::iterator it = PPFMap.find(ppf_);
			if (it == PPFMap.end())
				PPFMap.insert (std::pair<std::vector<float>, float>(ppf_,1));
			else{
				it->second = it->second + 1;
				if(it->second > max_val)
					max_val = it->second;
			}
		}
		max_count_ppf = max_val;
		std::cout << "PPFMap size is: " << PPFMap.size() << std::endl;
		
		system(("rm -rf " + std::string(argv[1]) + "/PPFMap.txt").c_str());
		system(("rm -rf " + std::string(argv[1]) + "/sampled_model.ply").c_str());

		std::map<std::vector<float>, float>::iterator it;
		for (it=PPFMap.begin(); it!=PPFMap.end(); ++it){
    		ofstream pFile;
		  	pFile.open ((std::string(argv[1]) + "/PPFMap.txt").c_str(), std::ofstream::out | std::ofstream::app);
		  	pFile << it->first[0] << " " << it->first[1] << " " << it->first[2] << " " << it->first[3] << " " << it->second << " " << max_val << '\n';
		  	pFile.close();
		}

		pcl::io::savePLYFile((std::string(argv[1]) + "/sampled_model.ply").c_str(), *cloud_sampled);

}