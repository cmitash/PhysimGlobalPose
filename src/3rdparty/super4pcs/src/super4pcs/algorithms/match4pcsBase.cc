// Copyright 2017 Nicolas Mellado
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// -------------------------------------------------------------------------- //
//
// Authors: Dror Aiger, Yoni Weill, Nicolas Mellado
//
// This file is part of the implementation of the 4-points Congruent Sets (4PCS)
// algorithm presented in:
//
// 4-points Congruent Sets for Robust Surface Registration
// Dror Aiger, Niloy J. Mitra, Daniel Cohen-Or
// ACM SIGGRAPH 2008 and ACM Transaction of Graphics.
//
// Given two sets of points in 3-space, P and Q, the algorithm applies RANSAC
// in roughly O(n^2) time instead of O(n^3) for standard RANSAC, using an
// efficient method based on invariants, to find the set of all 4-points in Q
// that can be matched by rigid transformation to a given set of 4-points in P
// called a base. This avoids the need to examine all sets of 3-points in Q
// against any base of 3-points in P as in standard RANSAC.
// The algorithm can use colors and normals to speed-up the matching
// and to improve the quality. It can be easily extended to affine/similarity
// transformation but then the speed-up is smaller because of the large number
// of congruent sets. The algorithm can also limit the range of transformations
// when the application knows something on the initial pose but this is not
// necessary in general (though can speed the runtime significantly).

// Home page of the 4PCS project (containing the paper, presentations and a
// demo): http://graphics.stanford.edu/~niloy/research/fpcs/fpcs_sig_08.html
// Use google search on "4-points congruent sets" to see many related papers
// and applications.

#include "match4pcsBase.h"
#include "shared4pcs.h"

#include <vector>
#include <chrono>
#include <queue>
#include <tuple>
#include <unordered_set>
#include <boost/functional/hash.hpp>

#include "Eigen/Core"
#include "Eigen/Geometry"                 // MatrixBase.homogeneous()
#include "Eigen/SVD"                      // Transform.computeRotationScaling()

#include "shared4pcs.h"
#include "sampling.h"
#include "accelerators/kdtree.h"

#include "io/io.h"
const double pi = std::acos(-1);

namespace std {
  template<typename... T>
  struct hash<tuple<T...>>
  {
      size_t operator()(tuple<T...> const& arg) const noexcept
      {
          return boost::hash_value(arg);
      }
  };
}

// Compute the closest points between two 3D line segments and obtain the two
// invariants corresponding to the closet points. This is the "intersection"
// point that determines the invariants. Since the 4 points are not exactly
// planar, we use the center of the line segment connecting the two closest
// points as the "intersection".
template < typename VectorType, typename Scalar>
static Scalar
distSegmentToSegment(const VectorType& p1, const VectorType& p2,
                     const VectorType& q1, const VectorType& q2,
                     Scalar& invariant1, Scalar& invariant2) {

  static const Scalar kSmallNumber = 0.0001;
  VectorType u = p2 - p1;
  VectorType v = q2 - q1;
  VectorType w = p1 - q1;
  Scalar a = u.dot(u);
  Scalar b = u.dot(v);
  Scalar c = v.dot(v);
  Scalar d = u.dot(w);
  Scalar e = v.dot(w);
  Scalar f = a * c - b * b;
  // s1,s2 and t1,t2 are the parametric representation of the intersection.
  // they will be the invariants at the end of this simple computation.
  Scalar s1 = 0.0;
  Scalar s2 = f;
  Scalar t1 = 0.0;
  Scalar t2 = f;

  if (f < kSmallNumber) {
    s1 = 0.0;
    s2 = 1.0;
    t1 = e;
    t2 = c;
  } else {
    s1 = (b * e - c * d);
    t1 = (a * e - b * d);
    if (s1 < 0.0) {
      s1 = 0.0;
      t1 = e;
      t2 = c;
    } else if (s1 > s2) {
      s1 = s2;
      t1 = e + b;
      t2 = c;
    }
  }

  if (t1 < 0.0) {
    t1 = 0.0;
    if (-d < 0.0)
      s1 = 0.0;
    else if (-d > a)
      s1 = s2;
    else {
      s1 = -d;
      s2 = a;
    }
  } else if (t1 > t2) {
    t1 = t2;
    if ((-d + b) < 0.0)
      s1 = 0;
    else if ((-d + b) > a)
      s1 = s2;
    else {
      s1 = (-d + b);
      s2 = a;
    }
  }
  invariant1 = (std::abs(s1) < kSmallNumber ? 0.0 : s1 / s2);
  invariant2 = (std::abs(t1) < kSmallNumber ? 0.0 : t1 / t2);

  return ( w + (invariant1 * u) - (invariant2 * v)).norm();
}

int approximate_bin(int val, int disc) {
  int lower_limit = val - (val % disc);
  int upper_limit = lower_limit + disc;

  int dist_from_lower = val - lower_limit;
  int dist_from_upper = upper_limit - val;

  int closest = (dist_from_lower < dist_from_upper)? lower_limit:upper_limit;

  return closest;
}

static void addToConnectivityMap(const std::vector<std::pair<int, int>>& pairs, float &distance,
                                std::map<std::pair<int, float>, std::vector<int> > &connectivity_map){

  for (size_t i = 0; i < pairs.size(); ++i) {
    std::pair<int, float> vertex_distance_pair = std::make_pair(pairs[i].first, distance);
    std::map<std::pair<int, float>, std::vector<int> >::iterator it = connectivity_map.find(vertex_distance_pair);

    if (it == connectivity_map.end()) {
      std::vector<int> vertices;
      vertices.push_back(pairs[i].second);
      connectivity_map.insert (std::make_pair(vertex_distance_pair,vertices));
    }
    else {
      it->second.push_back(pairs[i].second);
    }
  }

}

static void addToConnectivityPresenceMap(const std::vector<std::pair<int, int>>& pairs, float &distance,
                                std::map<std::pair<int, int>, int > &connectivity_map){

  for (size_t i = 0; i < pairs.size(); ++i) {
    std::map<std::pair<int, int>, int>::iterator it = connectivity_map.find(pairs[i]);
    if (it == connectivity_map.end())
      connectivity_map.insert (std::make_pair(pairs[i],1));
  }

}

static Eigen::Isometry3d convertToIsometry3d(Eigen::Matrix<float, 4, 4> &transformation){
  Eigen::Isometry3d poseIsometry;
  poseIsometry.setIdentity();

  for(int ii=0; ii<4; ii++)
    for(int jj=0; jj<4; jj++){
      poseIsometry.matrix()(ii,jj) = transformation(ii, jj);
    }

  return poseIsometry;
}

namespace Super4PCS{

Match4PCSBase::Match4PCSBase(const match_4pcs::Match4PCSOptions& options)
  : max_number_of_bases_(0),
    max_number_of_verifications_(0),
    max_base_diameter_(-1),
    P_mean_distance_(1.0),
    best_LCP_(0.0F),
    options_(options) {
  base_3D_.resize(4);
}

void Match4PCSBase::init(const std::vector<Point3D>& P,
                         const std::vector<Point3D>& Q,
                         const std::vector<Point3D>& Q_validation,
                         std::string probImagePath,
                         Eigen::Matrix3f camIntrinsic,
                         std::string objName,
                         std::map<std::vector<int>, std::vector<std::pair<int,int> > > &PPFMap, 
                         int max_count_ppf){

    start_time = clock();
    const Scalar kDiameterFraction = 0.3;

    centroid_P_ = VectorType::Zero();
    centroid_Q_ = VectorType::Zero();

    sampled_P_3D_.clear();
    sampled_Q_3D_.clear();
    validation_Q_3D.clear();

    sampled_P_3D_ = P;
    sampled_Q_3D_ = Q;
    validation_Q_3D = Q_validation;

    std::cout << "Super4PCS::Match4PCSBase::init:sampled_P_3D_size: " << sampled_P_3D_.size() << std::endl;
    std::cout << "Super4PCS::Match4PCSBase::init:sampled_Q_3D_size: " << sampled_Q_3D_.size() << std::endl;

    // Compute the centroids.
    for (int i = 0; i < sampled_P_3D_.size(); ++i) {
        centroid_P_ += sampled_P_3D_[i].pos();
    }

    centroid_P_ /= Scalar(sampled_P_3D_.size());

    for (int i = 0; i < sampled_Q_3D_.size(); ++i) {
        centroid_Q_ += sampled_Q_3D_[i].pos();
    }

    centroid_Q_ /= Scalar(sampled_Q_3D_.size());

    // Move the samples to the centroids to allow robustness in rotation.
    for (int i = 0; i < sampled_P_3D_.size(); ++i) {
        sampled_P_3D_[i].pos() -= centroid_P_;
    }
    for (int i = 0; i < sampled_Q_3D_.size(); ++i) {
        sampled_Q_3D_[i].pos() -= centroid_Q_;
    }
    for (int i = 0; i < validation_Q_3D.size(); ++i) {
        validation_Q_3D[i].pos() -= centroid_Q_;
    }

    for (int i = 0; i < hull_Q_3D.size(); ++i) {
        hull_Q_3D[i].pos() -= centroid_Q_;
    }

    initKdTree();
    // Compute the diameter of P approximately (randomly). This is far from being
    // Guaranteed close to the diameter but gives good results for most common
    // objects if they are densely sampled.
    P_diameter_ = 0.0;
    for (int i = 0; i < kNumberOfDiameterTrials; ++i) {
        int at = rand() % sampled_Q_3D_.size();
        int bt = rand() % sampled_Q_3D_.size();

        Scalar l = (sampled_Q_3D_[bt].pos() - sampled_Q_3D_[at].pos()).norm();
        if (l > P_diameter_) {
            P_diameter_ = l;
        }
    }

    // Normalize the delta (See the paper) and the maximum base distance.
    // delta = P_mean_distance_ * delta;
    max_base_diameter_ = P_diameter_;  // * estimated_overlap_;

    // maximum number of trails
    max_number_of_bases_ = 100;
    max_number_of_verifications_ = 100;

    base_selection_time = 0;
    congruent_set_extraction = 0;
    congruent_set_verification = 0;
    clustering_time = 0;
    total_time = 0;

    // setting the mode of operation, 0->super4pcs, 1->stoCS
    operMode = 1;

    // translation and rotation discretization for point pair features
    trans_disc = 5;
    rot_disc = 10;

    baseSet.clear();
    allTransforms.clear();

    current_trial_ = 0;
    best_LCP_ = 0.0;
    best_lcp_index = -1;

    // call Virtual handler
    Initialize(P, Q);

    // Reading the probability image
    cv::Mat probImgRaw = cv::imread(probImagePath, CV_16UC1);
    cv::Mat probImg = cv::Mat::zeros(probImgRaw.rows, probImgRaw.cols, CV_32FC1);
    for(int u=0; u<probImgRaw.rows; u++)
      for(int v=0; v<probImgRaw.cols; v++){
        unsigned short probShort = probImgRaw.at<unsigned short>(u,v);
        float prob = (float)probShort/10000;
        probImg.at<float>(u, v) = prob;
      }

    // Priority based sampling
    for (int i = 0; i < sampled_P_3D_.size(); ++i) {
      Point3D b_ii  = sampled_P_3D_[i];
      b_ii.pos() += centroid_P_;
      double x1 = b_ii.x();
      double y1 = b_ii.y();
      double z1 = b_ii.z();

      Eigen::Vector3f point2D = camIntrinsic * Eigen::Vector3f(x1, y1, z1);
      int col = point2D[0]/point2D[2];
      int row = point2D[1]/point2D[2];

      orig_probabilities_.push_back(probImg.at<float>(row, col));
      corr_pixels.push_back(std::make_pair(row,col));
    }

    this->registered_indices.clear();
    this->PPFMap = &PPFMap;
    this->max_count_ppf = max_count_ppf;
}

void
Match4PCSBase::getRegisteredModel(const Eigen::Ref<const MatrixType> &mat) {

  // We allow factor 2 scaling in the normalization.
  const Scalar epsilon = options_.delta;

  const size_t number_of_points = sampled_Q_3D_.size();

  const Scalar sq_eps = epsilon*epsilon;

  for (int i = 0; i < number_of_points; ++i) {

    // Use the kdtree to get the nearest neighbor
    Super4PCS::KdTree<Scalar>::Index resId =
    kd_tree_.doQueryRestrictedClosestIndex(
                (mat * sampled_Q_3D_[i].pos().homogeneous()).head<3>(),
                sq_eps);

    if ( resId != Super4PCS::KdTree<Scalar>::invalidIndex() ) {

        VectorType n_q = mat.block<3,3>(0,0)*sampled_Q_3D_[i].normal();
        float angle_n = std::acos(sampled_P_3D_[resId].normal().dot(n_q))*180/M_PI;
        // angle_n = std::min(angle_n, fabs(180-angle_n));
        if(angle_n < 30){
          registered_indices.push_back(resId);
        }
    }
  }
}

bool Match4PCSBase::SelectRandomTriangle(int &base1, int &base2, int &base3) {
      int number_of_points = sampled_P_3D_.size();
      base1 = base2 = base3 = -1;

      // Pick the first point at random.
      int first_point = rand() % number_of_points;

      const Scalar sq_max_base_diameter_ = max_base_diameter_*max_base_diameter_;

      // Try fixed number of times retaining the best other two.
      Scalar best_wide = 0.0;
      for (int i = 0; i < kNumberOfDiameterTrials; ++i) {
        // Pick and compute
        const int second_point = rand() % number_of_points;
        const int third_point = rand() % number_of_points;
        const VectorType u =
                sampled_P_3D_[second_point].pos() -
                sampled_P_3D_[first_point].pos();
        const VectorType w =
                sampled_P_3D_[third_point].pos() -
                sampled_P_3D_[first_point].pos();
        // We try to have wide triangles but still not too large.
        Scalar how_wide = (u.cross(w)).norm();
        if (how_wide > best_wide &&
                u.squaredNorm() < sq_max_base_diameter_ &&
                w.squaredNorm() < sq_max_base_diameter_) {
          best_wide = how_wide;
          base1 = first_point;
          base2 = second_point;
          base3 = third_point;
        }
      }
      return base1 != -1 && base2 != -1 && base3 != -1;
}

// Try the current base in P and obtain the best pairing, i.e. the one that
// gives the smaller distance between the two closest points. The invariants
// corresponding the the base pairing are computed.
bool Match4PCSBase::TryQuadrilateral(Scalar &invariant1, Scalar &invariant2,
                                     int& id1, int& id2, int& id3, int& id4) {

  Scalar min_distance = std::numeric_limits<Scalar>::max();
  int best1, best2, best3, best4;
  best1 = best2 = best3 = best4 = -1;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      if (i == j) continue;
      int k = 0;
      while (k == i || k == j) k++;
      int l = 0;
      while (l == i || l == j || l == k) l++;
      double local_invariant1;
      double local_invariant2;
      // Compute the closest points on both segments, the corresponding
      // invariants and the distance between the closest points.
      Scalar segment_distance = distSegmentToSegment(
                  base_3D_[i].pos(), base_3D_[j].pos(),
                  base_3D_[k].pos(), base_3D_[l].pos(),
                  local_invariant1, local_invariant2);
      // Retail the smallest distance and the best order so far.
      if (segment_distance < min_distance) {
        min_distance = segment_distance;
        best1 = i;
        best2 = j;
        best3 = k;
        best4 = l;
        invariant1 = local_invariant1;
        invariant2 = local_invariant2;
      }
    }
  }

  if(best1 < 0 || best2 < 0 || best3 < 0 || best4 < 0 ) return false;

  std::vector<Point3D> tmp = base_3D_;
  base_3D_[0] = tmp[best1];
  base_3D_[1] = tmp[best2];
  base_3D_[2] = tmp[best3];
  base_3D_[3] = tmp[best4];

  std::array<int, 4> tmpId = {id1, id2, id3, id4};
  id1 = tmpId[best1];
  id2 = tmpId[best2];
  id3 = tmpId[best3];
  id4 = tmpId[best4];

  return true;
}

bool Match4PCSBase::SelectTetrahedronBase(Scalar& invariant1, Scalar& invariant2,
                                        int& base1, int& base2, int& base3,
                                        int& base4) {
  int current_trial = 0;
  const Scalar kBaseTooSmall (0.2);
  const Scalar too_small = std::pow(max_base_diameter_ * kBaseTooSmall, 2);
  base1 = base2 = base3 = base4 = -1;

  while (current_trial < kNumberOfDiameterTrials) {
    // Select a triangle if possible. otherwise fail.
    if (!SelectRandomTriangle(base1, base2, base3)){
      return false;
    }

    float max_volume = 0;
    for (int ii=0; ii< 100; ii++){
      int number_of_points = sampled_P_3D_.size();
      int fourth_point = rand() % number_of_points;
      const Scalar sq_max_base_diameter_ = max_base_diameter_*max_base_diameter_;

      const VectorType v1 = sampled_P_3D_[base2].pos() - sampled_P_3D_[base1].pos();
      const VectorType v2 = sampled_P_3D_[base3].pos() - sampled_P_3D_[base1].pos();
      const VectorType v3 = sampled_P_3D_[fourth_point].pos() - sampled_P_3D_[base1].pos();

      float vol = abs((v1.cross(v2)).dot(v3))/6;

      if (vol > max_volume) {
        base4 = fourth_point;
        max_volume = vol;
      }
    }

    if(base4 != -1) return true;
    current_trial++;
  }

  return false;
}

// Selects a good base from P and computes its invariants. Returns false if
// a good planar base cannot can be found.
bool Match4PCSBase::SelectQuadrilateral(Scalar& invariant1, Scalar& invariant2,
                                        int& base1, int& base2, int& base3,
                                        int& base4) {

  const Scalar kBaseTooSmall (0.1);
  int current_trial = 0;

  // Try fix number of times.
  while (current_trial < kNumberOfDiameterTrials) {
    // Select a triangle if possible. otherwise fail.
    if (!SelectRandomTriangle(base1, base2, base3)){
      return false;
    }

    base_3D_[0] = sampled_P_3D_[base1];
    base_3D_[1] = sampled_P_3D_[base2];
    base_3D_[2] = sampled_P_3D_[base3];

    // The 4th point will be a one that is close to be planar to the other 3
    // while still not too close to them.
    const double x1 = base_3D_[0].x();
    const double y1 = base_3D_[0].y();
    const double z1 = base_3D_[0].z();
    const double x2 = base_3D_[1].x();
    const double y2 = base_3D_[1].y();
    const double z2 = base_3D_[1].z();
    const double x3 = base_3D_[2].x();
    const double y3 = base_3D_[2].y();
    const double z3 = base_3D_[2].z();

    // Fit a plan.
    Scalar denom = (-x3 * y2 * z1 + x2 * y3 * z1 + x3 * y1 * z2 - x1 * y3 * z2 -
                    x2 * y1 * z3 + x1 * y2 * z3);

    if (denom != 0) {
      Scalar A =
          (-y2 * z1 + y3 * z1 + y1 * z2 - y3 * z2 - y1 * z3 + y2 * z3) / denom;
      Scalar B =
          (x2 * z1 - x3 * z1 - x1 * z2 + x3 * z2 + x1 * z3 - x2 * z3) / denom;
      Scalar C =
          (-x2 * y1 + x3 * y1 + x1 * y2 - x3 * y2 - x1 * y3 + x2 * y3) / denom;
      base4 = -1;
      Scalar best_distance = std::numeric_limits<Scalar>::max();
      // Go over all points in P.
      const Scalar too_small = std::pow(max_base_diameter_ * kBaseTooSmall, 2);
      for (unsigned int i = 0; i < sampled_P_3D_.size(); ++i) {
        if ((sampled_P_3D_[i].pos()- sampled_P_3D_[base1].pos()).squaredNorm() >= too_small &&
            (sampled_P_3D_[i].pos()- sampled_P_3D_[base2].pos()).squaredNorm() >= too_small &&
            (sampled_P_3D_[i].pos()- sampled_P_3D_[base3].pos()).squaredNorm() >= too_small) {
          // Not too close to any of the first 3.
          const Scalar distance =
              std::abs(A * sampled_P_3D_[i].x() + B * sampled_P_3D_[i].y() +
                   C * sampled_P_3D_[i].z() - 1.0);
          // Search for the most planar.
          if (distance < best_distance) {
            best_distance = distance;
            base4 = int(i);
          }
        }
      }
      // std::cout << "best planar distance: " << best_distance << std::endl;
      // If we have a good one we can quit.
      if (base4 != -1) {
        base_3D_[3] = sampled_P_3D_[base4];
        if(TryQuadrilateral(invariant1, invariant2, base1, base2, base3, base4))
            return true;
      }
    }
    current_trial++;
  }

  // We failed to find good enough base..
  return false;
}

bool Match4PCSBase::computePPF(int &pIdx1, int &pIdx2, std::vector<int> &ppf_) {
  VectorType p1 = sampled_P_3D_[pIdx1].pos();
  VectorType p2 = sampled_P_3D_[pIdx2].pos();
  VectorType n1 = sampled_P_3D_[pIdx1].normal();
  VectorType n2 = sampled_P_3D_[pIdx2].normal();
  VectorType u = p1 - p2;

  int ppf_1 = int(u.norm()*1000);
  int ppf_2 = int(atan2(n1.cross(u).norm(), n1.dot(u))*180/M_PI);
  int ppf_3 = int(atan2(n2.cross(u).norm(), n2.dot(u))*180/M_PI);
  int ppf_4 = int(atan2(n1.cross(n2).norm(), n1.dot(n2))*180/M_PI);

  ppf_.push_back(approximate_bin(ppf_1, trans_disc));
  ppf_.push_back(approximate_bin(ppf_2, rot_disc));
  ppf_.push_back(approximate_bin(ppf_3, rot_disc));
  ppf_.push_back(approximate_bin(ppf_4, rot_disc));
}

bool Match4PCSBase::SelectQuadrilateralStoCS(Scalar& invariant1, Scalar& invariant2,
                                        int& base1, int& base2, int& base3,
                                        int& base4, float& baseProbability) {
  ofstream pFile;
  std::vector<int> ppf_;
  bool point_present;
  float sum_probabilities;

  std::vector<float> curr_probabilities_(orig_probabilities_);
  
  // Select point 1
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator (seed);
  std::discrete_distribution<int> p_dist_1 (curr_probabilities_.begin(),curr_probabilities_.end());
  base1 = p_dist_1(generator);
  baseProbability = curr_probabilities_[base1];

  // for (int i = 0; i < curr_probabilities_.size(); i++) {
  //   ofstream pFile1;
  //   pFile1.open ("/home/chaitanya/Desktop/b1.txt", std::ofstream::out | std::ofstream::app);
  //   pFile1 << corr_pixels[i].first << " " << corr_pixels[i].second << " " << curr_probabilities_[i] << std::endl;
  //   pFile1.close();
  // }

  // Select point 2
  sum_probabilities = 0;
  point_present = false;
  for (int i = 0; i < sampled_P_3D_.size(); i++) {
    if(i == base1 || curr_probabilities_[i] == 0) {
      curr_probabilities_[i] = 0;
      continue;
    }

    // computing edge factor
    ppf_.clear();
    computePPF(base1, i, ppf_);
    auto it = PPFMap->find(ppf_);
    float edge_i_0 = (it == PPFMap->end()) ? 0:1;

    curr_probabilities_[i] = orig_probabilities_[i]*
      orig_probabilities_[base1]*
      edge_i_0;

    if(curr_probabilities_[i] != 0) point_present = true;

    sum_probabilities += curr_probabilities_[i];
  }
  if(point_present == false) return false;
  
  // update the probability distribution
  for (int i = 0; i < sampled_P_3D_.size(); i++)
    curr_probabilities_[i] /= sum_probabilities;
  
  std::discrete_distribution<int> p_dist_2 (curr_probabilities_.begin(), curr_probabilities_.end());
  base2 = p_dist_2(generator);
  baseProbability = baseProbability*curr_probabilities_[base2];

  // for (int i = 0; i < curr_probabilities_.size(); i++) {
  //   ofstream pFile1;
  //   pFile1.open ("/home/chaitanya/Desktop/b2.txt", std::ofstream::out | std::ofstream::app);
  //   pFile1 << corr_pixels[i].first << " " << corr_pixels[i].second << " " << curr_probabilities_[i] << std::endl;
  //   pFile1.close();
  // }

  // Select point 3
  sum_probabilities = 0;
  point_present = false;
  VectorType v_1 = sampled_P_3D_[base2].pos() - sampled_P_3D_[base1].pos();
  for (int i = 0; i < sampled_P_3D_.size(); i++) {
    VectorType v_2 = sampled_P_3D_[i].pos() - sampled_P_3D_[base1].pos();
    float int_angle = acos(v_1.dot(v_2))*180/M_PI;
    int_angle = std::min(int_angle, 180-int_angle);

    if(i == base1 || i == base2 || curr_probabilities_[i] == 0 || int_angle < 30){
      curr_probabilities_[i] = 0;
      continue;
    }

    // computing edge factor
    ppf_.clear();
    computePPF(base2, i, ppf_);

    auto it = PPFMap->find(ppf_);
    float edge_i_1 = (it == PPFMap->end()) ? 0:1;

    curr_probabilities_[i] = curr_probabilities_[i]*
      orig_probabilities_[base2]*
      edge_i_1;

    if(curr_probabilities_[i] != 0) point_present = true;
    sum_probabilities += curr_probabilities_[i];
  }
  if(point_present == false) return false;

  // update the probability distribution
  for (int i = 0; i < sampled_P_3D_.size(); i++)
    curr_probabilities_[i] /= sum_probabilities;

  std::discrete_distribution<int> p_dist_3 (curr_probabilities_.begin(), curr_probabilities_.end());
  base3 = p_dist_3(generator);
  baseProbability = baseProbability*curr_probabilities_[base3];

  // for (int i = 0; i < curr_probabilities_.size(); i++) {
  //   ofstream pFile1;
  //   pFile1.open ("/home/chaitanya/Desktop/b3.txt", std::ofstream::out | std::ofstream::app);
  //   pFile1 << corr_pixels[i].first << " " << corr_pixels[i].second << " " << curr_probabilities_[i] << std::endl;
  //   pFile1.close();
  // }
  
  // Select point 4
  sum_probabilities = 0;
  point_present = false;
  for (int i = 0; i < sampled_P_3D_.size(); i++) {
    if(i == base1 || i == base2 || i == base3 || curr_probabilities_[i] == 0){
      curr_probabilities_[i] = 0;
      continue;
    }

    // The 4th point will be a one that is close to be planar
    double x1 = sampled_P_3D_[base1].x();
    double y1 = sampled_P_3D_[base1].y();
    double z1 = sampled_P_3D_[base1].z();
    double x2 = sampled_P_3D_[base2].x();
    double y2 = sampled_P_3D_[base2].y();
    double z2 = sampled_P_3D_[base2].z();
    double x3 = sampled_P_3D_[base3].x();
    double y3 = sampled_P_3D_[base3].y();
    double z3 = sampled_P_3D_[base3].z();

    // Fit a plane
    Scalar denom = (-x3 * y2 * z1 + x2 * y3 * z1 + x3 * y1 * z2 - x1 * y3 * z2 -
                    x2 * y1 * z3 + x1 * y2 * z3);

    if (denom != 0) {
      Scalar A =
          (-y2 * z1 + y3 * z1 + y1 * z2 - y3 * z2 - y1 * z3 + y2 * z3) / denom;
      Scalar B =
          (x2 * z1 - x3 * z1 - x1 * z2 + x3 * z2 + x1 * z3 - x2 * z3) / denom;
      Scalar C =
          (-x2 * y1 + x3 * y1 + x1 * y2 - x3 * y2 - x1 * y3 + x2 * y3) / denom;

      Scalar planar_distance = std::abs(A * sampled_P_3D_[i].x() + B * sampled_P_3D_[i].y() +
        C * sampled_P_3D_[i].z() - 1.0);

      if(planar_distance > 0.01 ||
        (sampled_P_3D_[i].pos()- sampled_P_3D_[base1].pos()).norm() < 0.01 || 
        (sampled_P_3D_[i].pos()- sampled_P_3D_[base2].pos()).norm() < 0.01 || 
        (sampled_P_3D_[i].pos()- sampled_P_3D_[base3].pos()).norm() < 0.01 ) {
        curr_probabilities_[i] = 0;
        continue;  
      }
    }

    // computing edge factor
    ppf_.clear();
    computePPF(base3, i, ppf_);
    auto it = PPFMap->find(ppf_);
    float edge_i_2 = (it == PPFMap->end()) ? 0:1;

    curr_probabilities_[i] =  curr_probabilities_[i]*
      orig_probabilities_[base3]*
      edge_i_2;

    if(curr_probabilities_[i] != 0) point_present = true;
    sum_probabilities += curr_probabilities_[i];
  }

  if(point_present == false) return false;

  // update the probability distribution
  for (int i = 0; i < sampled_P_3D_.size(); i++)
    curr_probabilities_[i] /= sum_probabilities;

  std::discrete_distribution<int> p_dist_4 (curr_probabilities_.begin(),curr_probabilities_.end());
  base4 = p_dist_4(generator);
  baseProbability = baseProbability*curr_probabilities_[base4];

  // for (int i = 0; i < curr_probabilities_.size(); i++) {
  //   ofstream pFile1;
  //   pFile1.open ("/home/chaitanya/Desktop/b4.txt", std::ofstream::out | std::ofstream::app);
  //   pFile1 << corr_pixels[i].first << " " << corr_pixels[i].second << " " << curr_probabilities_[i] << std::endl;
  //   pFile1.close();
  // }

  base_3D_[0] = sampled_P_3D_[base1];
  base_3D_[1] = sampled_P_3D_[base2];
  base_3D_[2] = sampled_P_3D_[base3];
  base_3D_[3] = sampled_P_3D_[base4];

  TryQuadrilateral(invariant1, invariant2, base1, base2, base3, base4);

  return true;
}

/* ############################################################################################
################################ StoCS Voting #################################################
#############################################################################################*/

bool Match4PCSBase::SelectQuadrilateralStoCSVoting(Scalar& invariant1, Scalar& invariant2,
                                        int& base1, int& base2, int& base3,
                                        int& base4, float& baseProbability, int first_point_index) {
  ofstream pFile;
  std::vector<int> ppf_;
  bool point_present;
  float sum_probabilities;

  std::vector<float> curr_probabilities_(orig_probabilities_);
  
  // Select point 1
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator (seed);

  base1 = first_point_index;
  baseProbability = curr_probabilities_[base1];

  // Select point 2
  sum_probabilities = 0;
  point_present = false;
  for (int i = 0; i < sampled_P_3D_.size(); i++) {
    if(i == base1 || curr_probabilities_[i] == 0) {
      curr_probabilities_[i] = 0;
      continue;
    }

    // computing edge factor
    ppf_.clear();
    computePPF(base1, i, ppf_);
    auto it = PPFMap->find(ppf_);
    float edge_i_0 = (it == PPFMap->end()) ? 0:1;

    curr_probabilities_[i] = orig_probabilities_[i]*
      orig_probabilities_[base1]*
      edge_i_0;

    if(curr_probabilities_[i] != 0) point_present = true;

    sum_probabilities += curr_probabilities_[i];
  }
  if(point_present == false) return false;
  
  // update the probability distribution
  for (int i = 0; i < sampled_P_3D_.size(); i++)
    curr_probabilities_[i] /= sum_probabilities;
  
  std::discrete_distribution<int> p_dist_2 (curr_probabilities_.begin(), curr_probabilities_.end());
  base2 = p_dist_2(generator);
  baseProbability = baseProbability*curr_probabilities_[base2];

  // Select point 3
  sum_probabilities = 0;
  point_present = false;
  VectorType v_1 = sampled_P_3D_[base2].pos() - sampled_P_3D_[base1].pos();
  for (int i = 0; i < sampled_P_3D_.size(); i++) {
    VectorType v_2 = sampled_P_3D_[i].pos() - sampled_P_3D_[base1].pos();
    float int_angle = acos(v_1.dot(v_2))*180/M_PI;
    int_angle = std::min(int_angle, 180-int_angle);

    if(i == base1 || i == base2 || curr_probabilities_[i] == 0 || int_angle < 30){
      curr_probabilities_[i] = 0;
      continue;
    }

    // computing edge factor
    ppf_.clear();
    computePPF(base2, i, ppf_);

    auto it = PPFMap->find(ppf_);
    float edge_i_1 = (it == PPFMap->end()) ? 0:1;

    curr_probabilities_[i] = curr_probabilities_[i]*
      orig_probabilities_[base2]*
      edge_i_1;

    if(curr_probabilities_[i] != 0) point_present = true;
    sum_probabilities += curr_probabilities_[i];
  }
  if(point_present == false) return false;

  // update the probability distribution
  for (int i = 0; i < sampled_P_3D_.size(); i++)
    curr_probabilities_[i] /= sum_probabilities;

  std::discrete_distribution<int> p_dist_3 (curr_probabilities_.begin(), curr_probabilities_.end());
  base3 = p_dist_3(generator);
  baseProbability = baseProbability*curr_probabilities_[base3];

  // for (int i = 0; i < curr_probabilities_.size(); i++) {
  //   ofstream pFile1;
  //   pFile1.open ("/home/chaitanya/Desktop/b3.txt", std::ofstream::out | std::ofstream::app);
  //   pFile1 << corr_pixels[i].first << " " << corr_pixels[i].second << " " << curr_probabilities_[i] << std::endl;
  //   pFile1.close();
  // }
  
  // Select point 4
  sum_probabilities = 0;
  point_present = false;
  for (int i = 0; i < sampled_P_3D_.size(); i++) {
    if(i == base1 || i == base2 || i == base3 || curr_probabilities_[i] == 0){
      curr_probabilities_[i] = 0;
      continue;
    }

    // The 4th point will be a one that is close to be planar
    double x1 = sampled_P_3D_[base1].x();
    double y1 = sampled_P_3D_[base1].y();
    double z1 = sampled_P_3D_[base1].z();
    double x2 = sampled_P_3D_[base2].x();
    double y2 = sampled_P_3D_[base2].y();
    double z2 = sampled_P_3D_[base2].z();
    double x3 = sampled_P_3D_[base3].x();
    double y3 = sampled_P_3D_[base3].y();
    double z3 = sampled_P_3D_[base3].z();

    // Fit a plane
    Scalar denom = (-x3 * y2 * z1 + x2 * y3 * z1 + x3 * y1 * z2 - x1 * y3 * z2 -
                    x2 * y1 * z3 + x1 * y2 * z3);

    if (denom != 0) {
      Scalar A =
          (-y2 * z1 + y3 * z1 + y1 * z2 - y3 * z2 - y1 * z3 + y2 * z3) / denom;
      Scalar B =
          (x2 * z1 - x3 * z1 - x1 * z2 + x3 * z2 + x1 * z3 - x2 * z3) / denom;
      Scalar C =
          (-x2 * y1 + x3 * y1 + x1 * y2 - x3 * y2 - x1 * y3 + x2 * y3) / denom;

      Scalar planar_distance = std::abs(A * sampled_P_3D_[i].x() + B * sampled_P_3D_[i].y() +
        C * sampled_P_3D_[i].z() - 1.0);

      if(planar_distance > 0.01 ||
        (sampled_P_3D_[i].pos()- sampled_P_3D_[base1].pos()).norm() < 0.01 || 
        (sampled_P_3D_[i].pos()- sampled_P_3D_[base2].pos()).norm() < 0.01 || 
        (sampled_P_3D_[i].pos()- sampled_P_3D_[base3].pos()).norm() < 0.01 ) {
        curr_probabilities_[i] = 0;
        continue;  
      }
    }

    // computing edge factor
    ppf_.clear();
    computePPF(base3, i, ppf_);
    auto it = PPFMap->find(ppf_);
    float edge_i_2 = (it == PPFMap->end()) ? 0:1;

    curr_probabilities_[i] =  curr_probabilities_[i]*
      orig_probabilities_[base3]*
      edge_i_2;

    if(curr_probabilities_[i] != 0) point_present = true;
    sum_probabilities += curr_probabilities_[i];
  }

  if(point_present == false) return false;

  // update the probability distribution
  for (int i = 0; i < sampled_P_3D_.size(); i++)
    curr_probabilities_[i] /= sum_probabilities;

  std::discrete_distribution<int> p_dist_4 (curr_probabilities_.begin(),curr_probabilities_.end());
  base4 = p_dist_4(generator);
  baseProbability = baseProbability*curr_probabilities_[base4];

  base_3D_[0] = sampled_P_3D_[base1];
  base_3D_[1] = sampled_P_3D_[base2];
  base_3D_[2] = sampled_P_3D_[base3];
  base_3D_[3] = sampled_P_3D_[base4];

  Scalar segment_distance = distSegmentToSegment(
              base_3D_[0].pos(), base_3D_[1].pos(),
              base_3D_[2].pos(), base_3D_[3].pos(),
              invariant1, invariant2);

  return true;
}

/* ############################################################################################
################################ StoCS Voting Ends#############################################
#############################################################################################*/

bool Match4PCSBase::FindCongruentQuadrilateralsV4PCS(std::vector<std::pair<int, int>>& pairs1, 
                                                     std::vector<std::pair<int, int>>& pairs2,
                                                     std::vector<std::pair<int, int>>& pairs3,
                                                     std::vector<std::pair<int, int>>& pairs4,
                                                     std::vector<std::pair<int, int>>& pairs5,
                                                     std::vector<std::pair<int, int>>& pairs6,
                                                     float &distance1, float &distance2, float &distance3, 
                                                     float &distance4, float &distance5, float &distance6,
                                                     float &distance_threshold, std::vector<match_4pcs::Quadrilateral>* quadrilaterals) {
  if (quadrilaterals == nullptr) return false;
  
  std::map<std::pair<int, float>, std::vector<int> > connectivity_map_2, connectivity_map_3;
  addToConnectivityMap(pairs2, distance2, connectivity_map_2);
  addToConnectivityMap(pairs3, distance3, connectivity_map_3);

  std::map<std::pair<int, int>, int > connectivity_map_4, connectivity_map_5, connectivity_map_6;
  addToConnectivityPresenceMap(pairs4, distance4, connectivity_map_4);
  addToConnectivityPresenceMap(pairs5, distance5, connectivity_map_5);
  addToConnectivityPresenceMap(pairs6, distance6, connectivity_map_6);

  quadrilaterals->clear();
  std::unordered_set<std::tuple<int, int, int> > base_3;
  base_3.clear();

  for (size_t i = 0; i < pairs1.size(); ++i) {
    // search for v1, d2 in the connectivity map to find candidates for v3
    std::pair<int, Scalar> vertex_distance_pair = std::make_pair(pairs1[i].first, distance2);
    std::map<std::pair<int, float>, std::vector<int> >::iterator it = connectivity_map_2.find(vertex_distance_pair);

    if (it != connectivity_map_2.end()) {
      std::vector<int> vertices = it->second;
      for(int j=0; j < vertices.size(); j++){
        std::map<std::pair<int, int>, int >::iterator conn_it4 = connectivity_map_4.find(std::make_pair(vertices[j], pairs1[i].second));
        if (conn_it4 != connectivity_map_4.end())
          base_3.insert(std::make_tuple(pairs1[i].first, pairs1[i].second, vertices[j]));
      }
    }
  }

  // std::cout << "distances: " << distance1 << " " << distance2 << " " << distance3 << " " << distance4 << " " << distance5 << " " << distance6 << std::endl;
  // std::cout << "base 3 size: " << base_3.size() << std::endl;

  // search for v4
  std::unordered_set<std::tuple<int, int, int> >::iterator base3_it;
  for (base3_it = base_3.begin(); base3_it != base_3.end(); base3_it++){
    int v1 = std::get<0>(*base3_it);
    int v2 = std::get<1>(*base3_it);
    int v3 = std::get<2>(*base3_it);

    std::pair<int, Scalar> vertex_distance_pair = std::make_pair(v1, distance3);
    std::map<std::pair<int, float>, std::vector<int> >::iterator it = connectivity_map_3.find(vertex_distance_pair);

    if (it != connectivity_map_3.end()) {
      std::vector<int> vertices = it->second;
      for(int k=0; k < vertices.size(); k++){
        std::map<std::pair<int, int>, int >::iterator conn_it5 = connectivity_map_5.find(std::make_pair(vertices[k], v2));
        std::map<std::pair<int, int>, int >::iterator conn_it6 = connectivity_map_6.find(std::make_pair(vertices[k], v3));

        if ((conn_it5 != connectivity_map_5.end()) && (conn_it6 != connectivity_map_6.end()))
          quadrilaterals->emplace_back(v1, v2, v3, vertices[k]);
      }
    }

  } // iterate over triplets

  return true;
}

void Match4PCSBase::initKdTree(){
  size_t number_of_points = sampled_P_3D_.size();

  // Build the kdtree.
  kd_tree_ = Super4PCS::KdTree<Scalar>(number_of_points);

  for (size_t i = 0; i < number_of_points; ++i) {
    kd_tree_.add(sampled_P_3D_[i].pos());
  }
  kd_tree_.finalize();
}

/*################################################################################################
#################################### FOR HOUGH VOTING ############################################
################################################################################################*/

static inline void rtToPose(Eigen::Matrix3d& R, Eigen::Vector3d& t, Eigen::Matrix4d& Pose)
{
  Pose << R(0,0), R(0,1), R(0,2), t[0],
          R(1,0), R(1,1), R(1,2), t[1],
          R(2,0), R(2,1), R(2,2), t[2],
          0, 0, 0, 1;
}

static inline void aaToR(Eigen::Vector3d& axis, double angle, Eigen::Matrix3d& R)
{
  const double sinA = sin(angle);
  const double cosA = cos(angle);
  const double cos1A = (1 - cosA);
  uint i, j;

  R << cosA, 0, 0,
       0, cosA, 0,
       0, 0, cosA;

  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
    {
      if (i != j)
      {
        // Symmetry skew matrix
        R(i, j) += (((i + 1) % 3 == j) ? -1 : 1) * sinA * axis[3 - i - j];

      }
      R(i, j) += cos1A * axis[i] * axis[j];
    }

}

static inline void TNormalize3(Eigen::Vector3d& v)
{
  double norm = v.norm();
  if (norm > EPS)
  {
    v *= 1.0 / norm;
  }
}

void Match4PCSBase::computeTransformRT(VectorType& p1_t, const VectorType& n1_t, Eigen::Matrix3d& R, Eigen::Vector3d& t)
{
  Eigen::Vector3d p1,n1;
  p1 << p1_t[0], p1_t[1], p1_t[2];
  n1 << n1_t[0], n1_t[1], n1_t[2];

  // dot product with x axis
  double angle = acos(n1[0]);

  // cross product with x axis
  Eigen::Vector3d axis(0, n1[2], -n1[1]);

  // we try to project on the ground plane but it's already parallel
  if (n1[1] == 0 && n1[2] == 0)
  {
    axis[1] = 1;
    axis[2] = 0;
  }
  else
  {
    TNormalize3(axis);
  }

  // std::cout << "axis: " << axis << std::endl;
  // std::cout << "angle: " << angle << std::endl;

  aaToR(axis, angle, R);
  t = -R * p1;
}

double Match4PCSBase::computeAlpha(VectorType& p1_t, const VectorType& n1_t, VectorType& p2_t)
{
  Eigen::Vector3d p1,n1,p2;
  p1 << p1_t[0], p1_t[1], p1_t[2];
  n1 << n1_t[0], n1_t[1], n1_t[2];
  p2 << p2_t[0], p2_t[1], p2_t[2];

  Eigen::Vector3d Tmg, mpt;
  Eigen::Matrix3d R;
  double alpha;
  
  computeTransformRT(p1_t, n1_t, R, Tmg);
  mpt = Tmg + R * p2;
  alpha=atan2(-mpt[2], mpt[1]);

  if ( alpha != alpha)
  {
    return 0;
  }

  if (sin(alpha)*mpt[2]<0.0)
    alpha=-alpha;

  return (-alpha);
}

/**
 * Compute a rotation in order to rotate around X direction
 */
static inline void getUnitXRotation(double angle, Eigen::Matrix3d& Rx)
{
  const double sx = sin(angle);
  const double cx = cos(angle);

  Rx << 1, 0, 0,
        0, cx, -sx,
        0, sx, cx;
}

bool Match4PCSBase::ComputeRigidTransformFromCongruentPairHough(std::vector<Super4PCS::BaseGraph*> &baseSet,
  std::vector< std::pair <Eigen::Isometry3d, float> > &allPose){ 

  float angle_step = (6.0)*M_PI/180.0;
  int numAlphaBins = (int) (floor (2 * M_PI / angle_step));
  int numModelPoints = sampled_Q_3D_.size();

  int refIndMax = 0;
  int alphaIndMax = 0;
  int maxVotes = 0;

  std::vector<int> accumulator(numModelPoints*numAlphaBins, 0);

  // the first point on the segment is the origin of the local reference frame
  Point3D s1;
  for (auto base_it: baseSet) {

    // get alpha_s for first 2 points
    s1 = sampled_P_3D_[base_it->baseIds_[0]];
    Point3D s2 = sampled_P_3D_[base_it->baseIds_[1]];
    
    double alpha_scene_1 = computeAlpha(s1.pos(), s1.normal(), s2.pos());

    for (int jj = 0; jj < base_it->congruent_quads.size(); jj++) {
      // for the first pair
      int model_index_1 = base_it->congruent_quads[jj].vertices[0];
      int model_index_2 = base_it->congruent_quads[jj].vertices[1];

      Point3D model_point_1 = sampled_Q_3D_[model_index_1];
      Point3D model_point_2 = sampled_Q_3D_[model_index_2];

      double alpha_model_1 = computeAlpha(model_point_1.pos(), model_point_1.normal(), model_point_2.pos());
      double alpha_1 = alpha_model_1 - alpha_scene_1;
      int alpha_index_1 = (int)(numAlphaBins*(alpha_1 + 2*M_PI) / (4*M_PI));
      accumulator[model_index_1*numAlphaBins + alpha_index_1]++;
    }
  }

  // Maximize the accumulator
  for (int k = 0; k < numModelPoints; k++)
  {
    for (int j = 0; j < numAlphaBins; j++)
    {
      const int accInd = k*numAlphaBins + j;
      const int accVal = accumulator[ accInd ];
      if (accVal > maxVotes)
      {
        maxVotes = accVal;
        refIndMax = k;
        alphaIndMax = j;
      }
    }
  }

  Point3D best_model_point;
  
  Eigen::Matrix4d Talpha;
  double alpha;

  Eigen::Vector3d tsg;
  Eigen::Matrix3d Rsg;
  Eigen::Matrix3d RInv;

  Eigen::Vector3d tmg;
  Eigen::Matrix3d Rmg;
  Eigen::Vector3d tInv;

  Eigen::Matrix4d TsgInv;
  Eigen::Matrix4d Tmg;

  computeTransformRT(s1.pos(), s1.normal(), Rsg, tsg);

  RInv = Rsg.transpose();
  tInv = -RInv * tsg;

  rtToPose(RInv, tInv, TsgInv);

  best_model_point = sampled_Q_3D_[refIndMax];

  computeTransformRT(best_model_point.pos(), best_model_point.normal(), Rmg, tmg);

  rtToPose(Rmg, tmg, Tmg);

  alpha = (alphaIndMax*(4*M_PI))/numAlphaBins-2*M_PI;

  Eigen::Matrix3d R;
  Eigen::Vector3d t;
  getUnitXRotation(alpha, R);
  rtToPose(R, t, Talpha);

  Eigen::Matrix4d rawPose = TsgInv * (Talpha * Tmg);

  Eigen::Matrix<float, 4, 4> transformation;
  transformation << rawPose(0,0), rawPose(0,1), rawPose(0,2), rawPose(0,3),
                    rawPose(1,0), rawPose(1,1), rawPose(1,2), rawPose(1,3),
                    rawPose(2,0), rawPose(2,1), rawPose(2,2), rawPose(2,3),
                    rawPose(3,0), rawPose(3,1), rawPose(3,2), rawPose(3,3);

  // std::cout << "rawPose: " << rawPose << std::endl;

  // The transformation has been computed between the two point clouds centered
  // at the origin, we need to recompute the translation to apply it to the original clouds
  {
      Eigen::Matrix<Scalar, 3,1> centroid_P,centroid_Q;
      centroid_P = centroid_P_;
      centroid_Q = centroid_Q_;

      Eigen::Matrix<Scalar, 3, 3> rot, scale;
      Eigen::Transform<Scalar, 3, Eigen::Affine> (transformation).computeRotationScaling(&rot, &scale);
      transformation.col(3) = (centroid_P - ( rot * scale * (centroid_Q))).homogeneous();
  }

  allPose.push_back(std::make_pair(convertToIsometry3d(transformation), 0));

  return true;
}

bool Match4PCSBase::ComputeRigidTransformFromPPF(int reference_point_index,
  std::vector< std::pair <Eigen::Isometry3d, float> > &allPose){ 

  float angle_step = (12.0)*M_PI/180.0;
  int numAlphaBins = (int) (floor (2 * M_PI / angle_step));
  int numModelPoints = sampled_Q_3D_.size();

  int refIndMax = 0;
  int alphaIndMax = 0;
  int maxVotes = 0;

  std::vector<int> accumulator(numModelPoints*numAlphaBins, 0);

  // the first point on the segment is the origin of the local reference frame
  Point3D s1 = sampled_P_3D_[reference_point_index];
  int total_computations = 0;
  for (int ii=0; ii<sampled_P_3D_.size(); ii++) {
    // get alpha_s
    Point3D s2 = sampled_P_3D_[ii];
    
    double alpha_scene_1 = computeAlpha(s1.pos(), s1.normal(), s2.pos());

    std::vector<std::pair<int, int>> pairs;
    std::vector<int> ppf_;

    computePPF(reference_point_index, ii, ppf_);
    pairs.clear();

    auto it = PPFMap->find(ppf_);
    if(it != PPFMap->end())
      pairs = it->second;

    // std::cout << "voting for pairs: " << pairs.size() << std::endl; 
    total_computations +=  pairs.size();
    for (int jj = 0; jj < pairs.size(); jj++) {
      // for the first pair
      int model_index_1 = pairs[jj].first;
      int model_index_2 = pairs[jj].second;

      Point3D model_point_1 = sampled_Q_3D_[model_index_1];
      Point3D model_point_2 = sampled_Q_3D_[model_index_2];

      double alpha_model_1 = computeAlpha(model_point_1.pos(), model_point_1.normal(), model_point_2.pos());
      double alpha_1 = alpha_model_1 - alpha_scene_1;
      int alpha_index_1 = (int)(numAlphaBins*(alpha_1 + 2*M_PI) / (4*M_PI));
      accumulator[model_index_1*numAlphaBins + alpha_index_1]++;
    }
  }

  // Maximize the accumulator
  for (int k = 0; k < numModelPoints; k++)
  {
    for (int j = 0; j < numAlphaBins; j++)
    {
      const int accInd = k*numAlphaBins + j;
      const int accVal = accumulator[ accInd ];
      if (accVal > maxVotes)
      {
        maxVotes = accVal;
        refIndMax = k;
        alphaIndMax = j;
      }
    }
  }

  std::cout << "numModelPoints: " << numModelPoints << std::endl;
  std::cout << "numAlphaBins: " << numAlphaBins << std::endl;
  std::cout << "total_computations: " << total_computations << std::endl;

  Point3D best_model_point;
  
  Eigen::Matrix4d Talpha;
  double alpha;

  Eigen::Vector3d tsg;
  Eigen::Matrix3d Rsg;
  Eigen::Matrix3d RInv;

  Eigen::Vector3d tmg;
  Eigen::Matrix3d Rmg;
  Eigen::Vector3d tInv;

  Eigen::Matrix4d TsgInv;
  Eigen::Matrix4d Tmg;

  computeTransformRT(s1.pos(), s1.normal(), Rsg, tsg);

  RInv = Rsg.transpose();
  tInv = -RInv * tsg;

  rtToPose(RInv, tInv, TsgInv);

  best_model_point = sampled_Q_3D_[refIndMax];

  computeTransformRT(best_model_point.pos(), best_model_point.normal(), Rmg, tmg);

  rtToPose(Rmg, tmg, Tmg);

  alpha = (alphaIndMax*(4*M_PI))/numAlphaBins-2*M_PI;

  Eigen::Matrix3d R;
  Eigen::Vector3d t;
  t << 0, 0, 0;
  getUnitXRotation(alpha, R);
  rtToPose(R, t, Talpha);

  Eigen::Matrix4d rawPose = TsgInv * (Talpha * Tmg);

  Eigen::Matrix<float, 4, 4> transformation;
  transformation << rawPose(0,0), rawPose(0,1), rawPose(0,2), rawPose(0,3),
                    rawPose(1,0), rawPose(1,1), rawPose(1,2), rawPose(1,3),
                    rawPose(2,0), rawPose(2,1), rawPose(2,2), rawPose(2,3),
                    rawPose(3,0), rawPose(3,1), rawPose(3,2), rawPose(3,3);

  std::cout << "rawPose: " << rawPose << std::endl;

  allPose.push_back(std::make_pair(convertToIsometry3d(transformation), maxVotes));

  return true;
}

bool Match4PCSBase::ComputeRigidTransformFromCongruentPair(
        int base_id1,
        int base_id2,
        int base_id3,
        int base_id4,
        match_4pcs::Quadrilateral &congruent_quad,
        std::vector< std::pair <Eigen::Isometry3d, float> > &allPose){

  std::array<std::pair<Point3D, Point3D>,4> congruent_points;

  // get references to the basis coordinates
  const Point3D& b1 = sampled_P_3D_[base_id1];
  const Point3D& b2 = sampled_P_3D_[base_id2];
  const Point3D& b3 = sampled_P_3D_[base_id3];
  const Point3D& b4 = sampled_P_3D_[base_id4];

  // Centroid of the basis, computed once and using only the three first points
  Eigen::Matrix<Scalar, 3, 1> centroid1 = (b1.pos() + b2.pos() + b3.pos()) / Scalar(3);

  // Centroid of the sets, computed in the loop using only the three first points
  Eigen::Matrix<Scalar, 3, 1> centroid2;

  // set the basis coordinates in the congruent quad array
  congruent_points[0].first = b1;
  congruent_points[1].first = b2;
  congruent_points[2].first = b3;
  congruent_points[3].first = b4;

  Eigen::Matrix<Scalar, 4, 4> transform;

  const int a = congruent_quad.vertices[0];
  const int b = congruent_quad.vertices[1];
  const int c = congruent_quad.vertices[2];
  const int d = congruent_quad.vertices[3];
  congruent_points[0].second = sampled_Q_3D_[a];
  congruent_points[1].second = sampled_Q_3D_[b];
  congruent_points[2].second = sampled_Q_3D_[c];
  congruent_points[3].second = sampled_Q_3D_[d];

  centroid2 = (congruent_points[0].second.pos() +
               congruent_points[1].second.pos() +
               congruent_points[2].second.pos()) / Scalar(3.);

  Scalar rms = -1;
  Scalar lcp = 0;

  const bool ok =
  ComputeRigidTransformation(congruent_points,   // input congruent quads
                             centroid1,          // input: basis centroid
                             centroid2,          // input: candidate quad centroid
                             options_.max_angle * pi / 180.0, // maximum per-dimension angle, check return value to detect invalid cases
                             transform,          // output: transformation
                             rms,                // output: rms error of the transformation between the basis and the congruent quad
                             false
                             );             // state: compute scale ratio ?

  if(ok && rms >= Scalar(0.)) {
    allTransforms.push_back(transform);

    Eigen::Matrix<float, 4, 4> transformation;
    transformation = transform;
    // The transformation has been computed between the two point clouds centered
    // at the origin, we need to recompute the translation to apply it to the original clouds
    {
        Eigen::Matrix<Scalar, 3,1> centroid_P,centroid_Q;
        centroid_P = centroid_P_;
        centroid_Q = centroid_Q_;

        Eigen::Matrix<Scalar, 3, 3> rot, scale;
        Eigen::Transform<Scalar, 3, Eigen::Affine> (transformation).computeRotationScaling(&rot, &scale);
        transformation.col(3) = (centroid1 + centroid_P - ( rot * scale * (centroid2 + centroid_Q))).homogeneous();
    }

    allPose.push_back(std::make_pair(convertToIsometry3d(transformation), lcp));
  }

  return true;
}

Match4PCSBase::Scalar 
Match4PCSBase::verifyRigidTransform(Eigen::Matrix<Scalar, 4, 4> transform, std::vector<int> &temp_registered_indices) {
  // Verify the rest of the points in Q against P.
    Scalar lcp = 0;
    if(operMode == 0)
      lcp = Verify(transform);
    else if(operMode == 1)
      lcp = WeightedVerify(transform, temp_registered_indices);
    else if(operMode == 2)
      lcp = Verify(transform);

    return lcp;
}

bool Match4PCSBase::ComputeRigidTransformation(const std::array< std::pair<Point3D, Point3D>,4>& pairs,
        const Eigen::Matrix<Scalar, 3, 1>& centroid1,
        Eigen::Matrix<Scalar, 3, 1> centroid2,
        Scalar max_angle,
        Eigen::Ref<MatrixType> transform,
        Scalar& rms_,
        bool computeScale ) {

  rms_ = kLargeNumber;

  if (pairs.size() == 0 || pairs.size() % 2 != 0)
      return false;


  Scalar kSmallNumber = 1e-6;

  // We only use the first 3 pairs. This simplifies the process considerably
  // because it is the planar case.

  const VectorType& p0 = pairs[0].first.pos();
  const VectorType& p1 = pairs[1].first.pos();
  const VectorType& p2 = pairs[2].first.pos();
        VectorType  q0 = pairs[0].second.pos();
        VectorType  q1 = pairs[1].second.pos();
        VectorType  q2 = pairs[2].second.pos();

  Scalar scaleEst (1.);

  VectorType vector_p1 = p1 - p0;
  if (vector_p1.squaredNorm() == 0) return kLargeNumber;
  vector_p1.normalize();
  VectorType vector_p2 = (p2 - p0) - ((p2 - p0).dot(vector_p1)) * vector_p1;
  if (vector_p2.squaredNorm() == 0) return kLargeNumber;
  vector_p2.normalize();
  VectorType vector_p3 = vector_p1.cross(vector_p2);

  VectorType vector_q1 = q1 - q0;
  if (vector_q1.squaredNorm() == 0) return kLargeNumber;
  vector_q1.normalize();
  VectorType vector_q2 = (q2 - q0) - ((q2 - q0).dot(vector_q1)) * vector_q1;
  if (vector_q2.squaredNorm() == 0) return kLargeNumber;
  vector_q2.normalize();
  VectorType vector_q3 = vector_q1.cross(vector_q2);

  Eigen::Matrix<Scalar, 3, 3> rotation = Eigen::Matrix<Scalar, 3, 3>::Identity();

  Eigen::Matrix<Scalar, 3, 3> rotate_p;
  rotate_p.row(0) = vector_p1;
  rotate_p.row(1) = vector_p2;
  rotate_p.row(2) = vector_p3;

  Eigen::Matrix<Scalar, 3, 3> rotate_q;
  rotate_q.row(0) = vector_q1;
  rotate_q.row(1) = vector_q2;
  rotate_q.row(2) = vector_q3;

  rotation = rotate_p.transpose() * rotate_q;

  // Discard singular solutions. The rotation should be orthogonal.
  if (((rotation * rotation).diagonal().array() - Scalar(1) > kSmallNumber).any())
      return false;

  //FIXME
  if (max_angle >= 0) {
      // Discard too large solutions (todo: lazy evaluation during boolean computation
      if (! (
                  std::abs(std::atan2(rotation(2, 1), rotation(2, 2)))
                  <= max_angle &&

                  std::abs(std::atan2(-rotation(2, 0),
                                      std::sqrt(std::pow(rotation(2, 1),2) +
                                                std::pow(rotation(2, 2),2))))
                  <= max_angle &&

                  std::abs(atan2(rotation(1, 0), rotation(0, 0)))
                  <= max_angle
             ))
          return false;
  }


  //FIXME
  // Compute rms and return it.
  rms_ = Scalar(0.0);
  {
      VectorType first, transformed;

      //cv::Mat first(3, 1, CV_64F), transformed;
      for (int i = 0; i < 3; ++i) {
          first = scaleEst*pairs[i].second.pos() - centroid2;
          transformed = rotation * first;
          rms_ += (transformed - pairs[i].first.pos() + centroid1).norm();
      }
  }

  rms_ /= Scalar(pairs.size());

  Eigen::Transform<Scalar, 3, Eigen::Affine> etrans (Eigen::Transform<Scalar, 3, Eigen::Affine>::Identity());

  // compute rotation and translation
  {
      etrans.scale(scaleEst);       // apply scale factor
      etrans.translate(centroid1);  // translation between quads
      etrans.rotate(rotation);           // rotate to align frames
      etrans.translate(-centroid2); // move to congruent quad frame

      transform = etrans.matrix();
  }

  return true;
}

float Match4PCSBase::c_dist_pose(int index_1, int index_2) {
  size_t number_of_points = hull_Q_3D.size();

  float max_distance = 0;
  for(int ii=0; ii<number_of_points; ii++){
    float min_distance = FLT_MAX;

    Eigen::Matrix<Scalar, 3, 1> p = (allTransforms[index_1]*hull_Q_3D[ii].pos().homogeneous()).head<3>();
    for(int jj=0; jj<number_of_points; jj++){
      Eigen::Matrix<Scalar, 3, 1> q = (allTransforms[index_2]*hull_Q_3D[jj].pos().homogeneous()).head<3>();
      float dist = (p - q).norm();
      if(dist < min_distance)
        min_distance = dist;
    }
    
    if(min_distance > max_distance)
      max_distance = min_distance;
  }

  return max_distance;
}

float Match4PCSBase::c_dist_pose_mean(int index_1, int index_2) {
  size_t number_of_points = hull_Q_3D.size();

  float mean_distance = 0;
  for(int ii=0; ii<number_of_points; ii++){
    float min_distance = FLT_MAX;

    Eigen::Matrix<Scalar, 3, 1> p = (allTransforms[index_1]*hull_Q_3D[ii].pos().homogeneous()).head<3>();
    for(int jj=0; jj<number_of_points; jj++){
      Eigen::Matrix<Scalar, 3, 1> q = (allTransforms[index_2]*hull_Q_3D[jj].pos().homogeneous()).head<3>();
      float dist = (p - q).norm();
      if(dist < min_distance)
        min_distance = dist;
    }
    mean_distance += min_distance;
  }

  return mean_distance;
}

float Match4PCSBase::c_dist(int index_1, int index_2) {
  float max_distance = 0;

  for(int ii=0; ii<4; ii++){
    float min_distance = FLT_MAX;
    
    for(int jj=0; jj<4; jj++) {
      float dist = (sampled_P_3D_[baseSet[index_1]->baseIds_[ii]].pos() - sampled_P_3D_[baseSet[index_2]->baseIds_[jj]].pos()).norm();
      if(dist < min_distance)
        min_distance = dist;
    }

    if(min_distance > max_distance)
      max_distance = min_distance;
  }

  return max_distance;
}

float Match4PCSBase::c_dist_mean(int index_1, int index_2) {
  float mean_distance = 0;

  for(int ii=0; ii<4; ii++){
    float min_distance = FLT_MAX;
    
    for(int jj=0; jj<4; jj++) {
      float dist = (sampled_P_3D_[baseSet[index_1]->baseIds_[ii]].pos() - sampled_P_3D_[baseSet[index_2]->baseIds_[jj]].pos()).norm();
      if(dist < min_distance)
        min_distance = dist;
    }

    mean_distance += min_distance;
  }

  return mean_distance;
}

// Verify a given transformation by computing the number of points in P at
// distance at most (normalized) delta from some point in Q. In the paper
// we describe randomized verification. We apply deterministic one here with
// early termination. It was found to be fast in practice.
Match4PCSBase::Scalar
Match4PCSBase::Verify(const Eigen::Ref<const MatrixType> &mat) {

  // We allow factor 2 scaling in the normalization.
  const Scalar epsilon = options_.delta;
  
  int good_points = 0;

  const size_t number_of_points = validation_Q_3D.size();
  const int terminate_value = best_LCP_ * number_of_points;

  const Scalar sq_eps = epsilon*epsilon;

  for (int i = 0; i < number_of_points; ++i) {

    // Use the kdtree to get the nearest neighbor
    Super4PCS::KdTree<Scalar>::Index resId =
    kd_tree_.doQueryRestrictedClosestIndex(
                (mat * validation_Q_3D[i].pos().homogeneous()).head<3>(),
                sq_eps);

    if ( resId != Super4PCS::KdTree<Scalar>::invalidIndex() )
        good_points++;

    // We can terminate if there is no longer chance to get better than the
    // current best LCP.
    if (number_of_points - i + good_points < terminate_value) {
      break;
    }
  }

  return Scalar(good_points) / Scalar(number_of_points);
}

Match4PCSBase::Scalar
Match4PCSBase::WeightedVerify(const Eigen::Ref<const MatrixType> &mat, std::vector<int> &temp_registered_indices) {

  // We allow factor 2 scaling in the normalization.
  const Scalar epsilon = options_.delta;
  
  float weighted_match = 0;

  const size_t number_of_points = validation_Q_3D.size();

  const Scalar sq_eps = epsilon*epsilon;

  for (int i = 0; i < number_of_points; ++i) {

    // Use the kdtree to get the nearest neighbor
    Super4PCS::KdTree<Scalar>::Index resId =
    kd_tree_.doQueryRestrictedClosestIndex(
                (mat * validation_Q_3D[i].pos().homogeneous()).head<3>(),
                sq_eps);

    if ( resId != Super4PCS::KdTree<Scalar>::invalidIndex() ) {

        VectorType n_q = mat.block<3,3>(0,0)*validation_Q_3D[i].normal();
        float angle_n = std::acos(sampled_P_3D_[resId].normal().dot(n_q))*180/M_PI;
        angle_n = std::min(angle_n, fabs(180-angle_n));
        if(angle_n < 30){
          weighted_match += orig_probabilities_[resId];
          temp_registered_indices.push_back(resId);
        }
    }
  }

  return weighted_match / Scalar(number_of_points);
}

// The main 4PCS function. Computes the best rigid transformation and transfoms
// Q toward P by this transformation.
Match4PCSBase::Scalar
Match4PCSBase::ComputeTransformation(const std::vector<Point3D>& P,
                                     std::vector<Point3D>* Q,
                                     std::vector<Point3D>* Q_validation,
                                     std::vector<Point3D>* Q_hull,
                                     Eigen::Isometry3d &bestPose, 
                                     std::vector< std::pair <Eigen::Isometry3d, float> > &allPose,
                                     std::string probImagePath, std::map<std::vector<int>, std::vector<std::pair<int,int> > > &PPFMap, int max_count_ppf, 
                                     Eigen::Matrix3f camIntrinsic, std::string objName, std::string scenePath, std::vector<int> &registered_points) {

  if (Q == nullptr) return kLargeNumber;

  hull_Q_3D = *Q_hull;
  init(P, *Q, *Q_validation, probImagePath, camIntrinsic, objName, PPFMap, max_count_ppf);

  Perform_N_steps(Q, allPose, scenePath, objName);

  if(best_lcp_index != -1){
    bestPose = allPose[best_lcp_index].first;
    registered_points = this->registered_indices;
  }
  else{
    std::cout << "returning identity" << std::endl;
    bestPose.matrix().setIdentity();
  }

  return best_LCP_;

  // Perform_Hough_Voting(Q, allPose, scenePath, objName);
  // bestPose = allPose[0].first;

  // return 0.5;
}

bool Match4PCSBase::Perform_Hough_Voting(std::vector<Point3D>* Q,
                                    std::vector< std::pair <Eigen::Isometry3d, float> > &allPose, 
                                    std::string scenePath, std::string objName) {

  if (Q == nullptr)
    return false;

  for(int reference_point_index = 0; reference_point_index < sampled_P_3D_.size(); reference_point_index++) {
    bool TrueFalse = (rand() % 100) < 20;
    if(TrueFalse) {
      std::cout << "reference point : " << reference_point_index << "\n\n\n"; 
      ComputeRigidTransformFromPPF(reference_point_index, allPose);
    }
  }
    
  return true;
}

// Performs N RANSAC iterations and compute the best transformation.
bool Match4PCSBase::Perform_N_steps(std::vector<Point3D>* Q,
                                    std::vector< std::pair <Eigen::Isometry3d, float> > &allPose, 
                                    std::string scenePath, std::string objName) {
  if (Q == nullptr)
    return false;

  // Step 1: Base Selection
  clock_t base_selection_start = clock();
  while(baseSet.size() < max_number_of_bases_) {
    Scalar invariant1, invariant2;
    std::vector<int> baseIdx(4,0);
    float baseProbability;

    bool selectedBase = false;
    if(operMode == 0)
      selectedBase = SelectQuadrilateral(invariant1, invariant2, baseIdx[0], baseIdx[1], baseIdx[2], baseIdx[3]);
    else if(operMode == 1)
      selectedBase = SelectQuadrilateralStoCS(invariant1, invariant2, baseIdx[0], baseIdx[1], baseIdx[2], baseIdx[3], baseProbability);
    else if(operMode == 2)
      selectedBase = SelectTetrahedronBase(invariant1, invariant2, baseIdx[0], baseIdx[1], baseIdx[2], baseIdx[3]);

    if(selectedBase) {
      BaseGraph *b_t = new BaseGraph(baseIdx, invariant1, invariant2, baseProbability);
      baseSet.push_back(b_t);
    }
  }
  
  std::cout << "Base set pool size: " << baseSet.size() << std::endl;
  base_selection_time = float( clock () - base_selection_start ) /  CLOCKS_PER_SEC;

  // Step 3: Congruent Set Extraction
  clock_t cse_start = clock();
  for (auto base_it: baseSet) {
    ExtractCongruentSet(base_it);
    
    int max_sampled_csets = 100;
    if(base_it->congruent_quads.size() < max_sampled_csets) {
      for (int jj = 0; jj < base_it->congruent_quads.size(); jj++)
      ComputeRigidTransformFromCongruentPair(base_it->baseIds_[0], base_it->baseIds_[1],
                                              base_it->baseIds_[2], base_it->baseIds_[3],
                                              base_it->congruent_quads[jj], allPose);
    }
    else {
      std::unordered_set<int> c_set_indices;
      while(c_set_indices.size() < max_sampled_csets)
        c_set_indices.insert(rand() % base_it->congruent_quads.size());

      for(auto c_set_it: c_set_indices)
        ComputeRigidTransformFromCongruentPair(base_it->baseIds_[0], base_it->baseIds_[1],
                                              base_it->baseIds_[2], base_it->baseIds_[3],
                                              base_it->congruent_quads[c_set_it], allPose);
    }

    base_it++;
  }
  congruent_set_extraction = float( clock () - cse_start ) /  CLOCKS_PER_SEC;
  
  std::cout << "Number of poses: " << allPose.size() << std::endl;

  std::vector<int> selected_indices;
  std::vector<float> selection_time;

  // Step 3: Congruent Set Verification
  clock_t verification_start = clock ();
  int pose_index = 0;
  for (auto transforms_it: allTransforms) {
    std::vector<int> temp_registered_indices;
    Scalar lcp = verifyRigidTransform(transforms_it, temp_registered_indices);
    if (lcp > best_LCP_) {
      best_LCP_  = lcp;
      best_lcp_index = pose_index;
      best_transform = transforms_it;
      selected_indices.push_back(best_lcp_index);
      selection_time.push_back(float( clock () - start_time ) /  CLOCKS_PER_SEC);
      registered_indices = temp_registered_indices;
    }
    allPose[pose_index].second = lcp;
    pose_index++;
  }

  auto temp_poses = allPose;

  allPose.clear();
  for(int ii =0; ii< selected_indices.size();ii++) {
    allPose.push_back(temp_poses[selected_indices[ii]]);
    
    ofstream pFile;
    pFile.open ((scenePath + "debug_super4PCS/" + objName + "_time.txt").c_str(),
        std::ofstream::out | std::ofstream::app);
    pFile << selection_time[ii] << std::endl;
    pFile.close();
  }

  congruent_set_verification = float( clock () - verification_start ) /  CLOCKS_PER_SEC;
  total_time = float( clock () - start_time ) /  CLOCKS_PER_SEC;

  ofstream pFile;
  pFile.open ("/media/chaitanya/DATADRIVE0/datasets/YCB_Video_Dataset/time.txt", std::ofstream::out | std::ofstream::app);
  pFile << total_time << " " << base_selection_time << " "
        << congruent_set_extraction << " " << congruent_set_verification
        << " " << allPose.size() << std::endl;
  pFile.close();

  return true;
}

bool Match4PCSBase::ExtractCongruentSet(Super4PCS::BaseGraph* baseIt) {

  Scalar invariant1, invariant2;
  int base_id1, base_id2, base_id3, base_id4;
  std::vector<std::pair<int, int>> pairs1, pairs2, pairs3, pairs4, pairs5, pairs6;
  std::vector<int> ppf_1, ppf_2, ppf_3, ppf_4, ppf_5, ppf_6;
  float distance1, distance2, distance3, distance4, distance5, distance6;
  float normal_angle1, normal_angle2, normal_angle3, normal_angle4, normal_angle5, normal_angle6;

  base_id1 = baseIt->baseIds_[0];
  base_id2 = baseIt->baseIds_[1];
  base_id3 = baseIt->baseIds_[2];
  base_id4 = baseIt->baseIds_[3];

  invariant1 = baseIt->invariant1_;
  invariant2 = baseIt->invariant2_;

  base_3D_[0] = sampled_P_3D_[base_id1];
  base_3D_[1] = sampled_P_3D_[base_id2];
  base_3D_[2] = sampled_P_3D_[base_id3];
  base_3D_[3] = sampled_P_3D_[base_id4];

  // Computes distance between pairs.
  distance1 = (base_3D_[0].pos()- base_3D_[1].pos()).norm();
  distance6 = (base_3D_[2].pos()- base_3D_[3].pos()).norm();

  // Compute normal angles.
  normal_angle1 = (base_3D_[0].normal() - base_3D_[1].normal()).norm();
  normal_angle6 = (base_3D_[2].normal() - base_3D_[3].normal()).norm();

  // computing point pair features
  computePPF(base_id1, base_id2, ppf_1);
  computePPF(base_id3, base_id4, ppf_6);

  if(operMode == 0 || operMode == 2) {
    ExtractPairs(distance1, normal_angle1, distance_factor * options_.delta, 0,
                    1, &pairs1, ppf_1);
    
    ExtractPairs(distance6, normal_angle6, distance_factor * options_.delta, 2,
                    3, &pairs6, ppf_6);
  }
  else {
    pairs1.clear();
    pairs6.clear();

    auto it_1 = PPFMap->find(ppf_1);
    if(it_1 != PPFMap->end())
      pairs1 = it_1->second;

    auto it_2 = PPFMap->find(ppf_6);
    if(it_2 != PPFMap->end())
      pairs6 = it_2->second;
  }

  if (operMode == 0 || operMode == 1){
    if (pairs1.size() == 0 || pairs6.size() == 0) {
      return false;
    }

    if (!FindCongruentQuadrilaterals(invariant1, invariant2,
                                     distance_factor * options_.delta,
                                     distance_factor * options_.delta,
                                     pairs1,
                                     pairs6,
                                     &baseIt->congruent_quads)) {
      return false;
    }
  }
  else {
    distance2 = (base_3D_[0].pos()- base_3D_[2].pos()).norm();
    distance3 = (base_3D_[0].pos()- base_3D_[3].pos()).norm();
    distance4 = (base_3D_[1].pos()- base_3D_[2].pos()).norm();
    distance5 = (base_3D_[1].pos()- base_3D_[3].pos()).norm();

    normal_angle2 = (base_3D_[0].normal() - base_3D_[2].normal()).norm();
    normal_angle3 = (base_3D_[0].normal() - base_3D_[3].normal()).norm();
    normal_angle4 = (base_3D_[1].normal() - base_3D_[2].normal()).norm();
    normal_angle5 = (base_3D_[1].normal() - base_3D_[3].normal()).norm();

    computePPF(base_id1, base_id3, ppf_2);
    computePPF(base_id1, base_id4, ppf_3);
    computePPF(base_id2, base_id3, ppf_4);
    computePPF(base_id2, base_id4, ppf_5);

    ExtractPairs(distance2, normal_angle2, distance_factor * options_.delta, 0,
                  2, &pairs2, ppf_2);
    ExtractPairs(distance3, normal_angle3, distance_factor * options_.delta, 0,
                    3, &pairs3, ppf_3);
    ExtractPairs(distance4, normal_angle4, distance_factor * options_.delta, 1,
                    2, &pairs4, ppf_4);
    ExtractPairs(distance5, normal_angle5, distance_factor * options_.delta, 1,
                    3, &pairs5, ppf_5);

    if (pairs1.size() == 0 || pairs2.size() == 0 ||pairs3.size() == 0 ||pairs4.size() == 0 ||pairs5.size() == 0 ||pairs6.size() == 0) {
      return false;
    }

    float distance_threshold = distance_factor * options_.delta;
    if (!FindCongruentQuadrilateralsV4PCS(pairs1, pairs2, pairs3, 
                                          pairs4, pairs5, pairs6,
                                          distance1, distance2, 
                                          distance3, distance4,
                                          distance5, distance6,
                                          distance_threshold, 
                                          &baseIt->congruent_quads)) {
      return false;
    }
  }

  return true;
}

BaseGraph::BaseGraph(std::vector<int> baseIds, float invariant1, float invariant2, float baseProbability){
  baseIds_[0] = baseIds[0];
  baseIds_[1] = baseIds[1];
  baseIds_[2] = baseIds[2];
  baseIds_[3] = baseIds[3];
  
  invariant1_ = invariant1;
  invariant2_ = invariant2;

  jointProbability = baseProbability;

  congruent_quads.clear();
}

} // namespace Super4PCS

