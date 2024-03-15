#include <boost/foreach.hpp>
#include <ctime>
#define foreach BOOST_FOREACH
#include <chrono>

#include "cylinder_fitting_hough.h"
#include "cylinder_fitting_ransac.h"
#include "helpers.h"
using namespace std;
using namespace std::chrono;

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

pcl::NormalEstimation<PointT, pcl::Normal> ne;
pcl::search::KdTree<PointT>::Ptr tree;

// $CYLINDER_TEST_EXEC $DATASET_DIR"cylinder/" $ANGLE_BINS $RADIUS_BINS
// $POSITION_BINS $ORIENTATION_ACCUMULATORS_NUM $GAUSSIAN_SPHERE_POINTS_NUM
// $ACCUMULATOR_PEAK_THRESHOLD $MIN_RADIUS $MAX_RADIUS $MODE $SOFT_VOTING
// $VISUALIZE ANGLE_BINS=30 RADIUS_BINS=20 POSITION_BINS=20
// ORIENTATION_ACCUMULATORS_NUM=30
// GAUSSIAN_SPHERE_POINTS_NUM=450
// ACCUMULATOR_PEAK_THRESHOLD=0.8
// MIN_RADIUS=0.02
// MAX_RADIUS=0.07
// MODE=1
// SOFT_VOTING=1
// VISUALIZE=1

int main(int argc, char** argv) {
  std::string pcd_file = argv[1];

  // HOUGH PARAMETERS
  unsigned int angle_bins = 360;
  unsigned int radius_bins = 10;
  unsigned int position_bins = 1000;

  // OUR HOUGH IMPLEMENTATION PARAMETERS
  unsigned int orientation_accumulators_num = 200;
  unsigned int gaussian_sphere_points_num = 3000;
  float accumulator_peak_threshold = 0.8;
  float min_radius = 0.025;
  float max_radius = 0.035;
  int mode = 1;
  bool soft_voting = true;  // paper is true!
  bool visualize = true;

  boost::shared_ptr<VisualizeFittingData> visualizer;

  if (visualize)
    visualizer =
        boost::shared_ptr<VisualizeFittingData>(new VisualizeFittingData());

  // Gaussian Sphere Uniform
  std::vector<double> weights;
  std::vector<Eigen::Matrix<double, 3, 1> > means;
  std::vector<Eigen::Matrix<double, 3, 1> > std_devs;
  weights.push_back(1.0);
  Eigen::Matrix<double, 3, 1> mean_eigen(0, 0, 0);
  means.push_back(mean_eigen);
  Eigen::Matrix<double, 3, 1> std_dev_eigen(0.5, 0.5, 0.5);
  std_devs.push_back(std_dev_eigen);

  GaussianMixtureModel gmm(weights, means, std_devs);
  GaussianSphere gaussian_sphere(gmm, gaussian_sphere_points_num,
                                 orientation_accumulators_num);

  boost::shared_ptr<CylinderFittingHough> cylinder_fitting(
      new CylinderFittingHough(
          gaussian_sphere, (unsigned int)angle_bins, (unsigned int)radius_bins,
          (unsigned int)position_bins, (float)min_radius, (float)max_radius,
          (float)accumulator_peak_threshold, mode, false, soft_voting));

  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
      new pcl::search::KdTree<pcl::PointXYZ>());

  pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(
      new pcl::PointCloud<pcl::Normal>());

  // Load 3D point cloud information in PCL format
  if (pcl::io::loadPCDFile<PointT>(pcd_file, *point_cloud)) {
    pcl::console::print_error("Error loading cloud file!\n");
    return (1);
  }

  // Normal estimation parameter (not organized)
  ne.setKSearch(6);
  ne.setSearchMethod(tree);
  ne.setInputCloud(point_cloud);
  ne.compute(*cloud_normals);
  FittingData fitting_data = cylinder_fitting->fit(point_cloud, cloud_normals);

  auto fitting_result = fitting_data.parameters;
  std::cout << fitting_result << std::endl;

  /* VISUALIZE */
  if (visualize) {
    auto viewer = fitting_data.visualize(point_cloud, visualizer);
    while (!viewer->wasStopped()) {
      viewer->spinOnce(100);
    }
  }
  /* END VISUALIZE */

  return (0);
}
