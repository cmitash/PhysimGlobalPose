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
double delta = 0.0035;

// Estimated overlap (see the paper).
double overlap = 0.5;

// Maximum norm of RGB values between corresponded points. 1e9 means don't use.
double max_color = -1;

// Number of sampled points in both files. The 4PCS allows a very aggressive sampling.
int n_points = 400;

// Maximum angle (degrees) between corresponded normals.
double norm_diff = 1;

// Maximum allowed computation time.
int max_time_seconds = 1;

bool use_super4pcs = true;

int getProbableTransformsSuper4PCS(std::string input1, std::string input2, 
      std::pair <Eigen::Isometry3d, float> &bestHypothesis, std::vector< std::pair <Eigen::Isometry3d, float> > &hypothesisSet) {
  
  using namespace Super4PCS;

  vector<Point3D> set1, set2;
  vector<Eigen::Matrix2f> tex_coords1, tex_coords2;
  vector<typename Point3D::VectorType> normals1, normals2;
  vector<tripple> tris1, tris2;
  vector<std::string> mtls1, mtls2;
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

  // clean only when we have pset to avoid wrong face to point indexation
  if (tris1.size() == 0)
    Utils::CleanInvalidNormals(set1, normals1);
  if (tris2.size() == 0)
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
    bestscore = matcher.ComputeTransformation(set1, &set2, bestPose, hypothesisSet);
  }
  catch (...) {
    std::cout << "[Unknown Error]: Aborting with code -3 ..." << std::endl;
    return -3;
  }

  bestHypothesis = std::make_pair(bestPose, bestscore);

  return 0;
}