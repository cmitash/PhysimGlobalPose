#include "algorithms/4pcs.h"
#include "algorithms/super4pcs.h"
#include "Eigen/Dense"

#include <fstream>
#include <iostream>
#include <string>

#include "io/io.h"
#include "utils/geometry.h"

#define sqr(x) ((x) * (x))

using namespace std;
using namespace match_4pcs;

//Parameters for the algorithm

// Delta (see the paper).
double delta = 0.005;

// Estimated overlap (see the paper).
double overlap = 0.5; // not used

// Maximum norm of RGB values between corresponded points. 1e9 means don't use.
double max_color = -1;

// Number of sampled points in both files. The 4PCS allows a very aggressive sampling.
int n_points = 400; // not used

// Maximum angle (degrees) between corresponded normals.
double norm_diff = 15;

// Maximum allowed computation time.
int max_time_seconds = 2;

bool use_super4pcs = true;

void getProbableTransformsSuper4PCS(std::string input1, std::string input2, std::string input3, 
      std::pair <Eigen::Isometry3d, float> &bestHypothesis, 
      std::vector< std::pair <Eigen::Isometry3d, float> > &hypothesisSet,
      std::string probImagePath, std::map<std::vector<int>, std::vector<std::pair<int,int> > > PPFMap, int max_count_ppf, 
      Eigen::Matrix3f camIntrinsic, std::string objName, std::string scenePath) {

  using namespace Super4PCS;

  vector<Point3D> set1, set2, set3, set4;
  vector<Eigen::Matrix2f> tex_coords1, tex_coords2, tex_coords3,tex_coords4;
  vector<typename Point3D::VectorType> normals1, normals2, normals3, normals4;
  vector<tripple> tris1, tris2, tris3,tris4;
  vector<std::string> mtls1, mtls2, mtls3,mtls4;
  Eigen::Isometry3d bestPose;
  float bestscore;

  IOManager iomananger;

  // Read the inputs.
  if (!iomananger.ReadObject((char *)input1.c_str(), set1, tex_coords1, normals1, tris1,
                  mtls1)) {
    perror("Can't read input set1");
    exit(-1);
  }

  if (!iomananger.ReadObject((char *)input2.c_str(), set2, tex_coords2, normals2, tris2,
                  mtls2)) {
    perror("Can't read input set2");
    exit(-1);
  }

  if (!iomananger.ReadObject((char *)input3.c_str(), set3, tex_coords3, normals3, tris3,
                  mtls3)) {
    perror("Can't read input set3");
    exit(-1);
  }

  if (!iomananger.ReadObject(("/home/chaitanya/github/PhysimGlobalPose/models/" + objName + "/hull.ply").c_str(), 
                  set4, tex_coords4, normals4, tris4, mtls4)) {
    perror("Can't read input set4");
    exit(-1);
  }

  // clean only when we have pset to avoid wrong face to point indexation
  if (tris1.size() == 0)
    Utils::CleanInvalidNormals(set1, normals1);
  if (tris2.size() == 0)
    Utils::CleanInvalidNormals(set2, normals2);
  if (tris3.size() == 0)
    Utils::CleanInvalidNormals(set2, normals2);

  // Our matcher.
  Match4PCSOptions options;

  // Set parameters.
  options.overlap_estimation = overlap;
  options.sample_size = n_points;
  options.max_normal_difference = norm_diff;
  options.max_color_distance = max_color;
  options.max_time_seconds = max_time_seconds;
  options.delta = delta;

  try {
    MatchSuper4PCS matcher(options);
    bestscore = matcher.ComputeTransformation(set1, &set3, &set2, &set4, bestPose, hypothesisSet, probImagePath, PPFMap, max_count_ppf, camIntrinsic, objName, scenePath);
  }
  catch (...) {
    std::cout << "[Unknown Error]: Aborting with code -3 ..." << std::endl;
  }

  bestHypothesis = std::make_pair(bestPose, bestscore);
}