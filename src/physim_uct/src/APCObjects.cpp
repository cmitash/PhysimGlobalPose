#include <APCObjects.hpp>

namespace apc_objects{
	std::map<std::string, int> objMap = boost::assign::map_list_of ("crayola_24_ct", 1)("expo_dry_erase_board_eraser", 2)("folgers_classic_roast_coffee", 3)
									 ("scotch_duct_tape", 4)("dasani_water_bottle", 5)("jane_eyre_dvd", 6)
									 ("up_glucose_bottle", 7)("laugh_out_loud_joke_book", 8)("soft_white_lightbulb", 9)
									 ("kleenex_tissue_box", 10)("ticonderoga_12_pencils", 11)("dove_beauty_bar", 12)
									 ("dr_browns_bottle_brush", 13)("elmers_washable_no_run_school_glue", 14)("rawlings_baseball", 15);
	std::string pathPclModels = "/models/pcl/";
	std::string pathObjModels = "/models/obj/";

	APCObjects::APCObjects(std::string name){
		objName = name;
		objIdx = objMap[name];
		pclModel = PointCloud::Ptr(new PointCloud);
		pcl::io::loadPCDFile(env_p + pathPclModels + name + ".pcd", *pclModel);
		pcl::io::loadPolygonFile(env_p + pathObjModels + name + ".obj" , objModel);

		pcl::VoxelGrid<pcl::PointXYZ> sor;
		sor.setInputCloud (pclModel);
		sor.setLeafSize (0.005f, 0.005f, 0.005f);
		sor.filter (*pclModel);
	}
}