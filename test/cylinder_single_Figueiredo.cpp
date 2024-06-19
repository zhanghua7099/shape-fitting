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
// ?????????使用自己的数据，中心不在圆柱轴附近就直接挂掉?????????

int main(int argc, char** argv) {
  // set the pcd path
  std::string pcd_file = argv[1];

  // HOUGH PARAMETERS
  unsigned int angle_bins = 30;
  unsigned int radius_bins = 10;
  unsigned int position_bins = 100;

  // OUR HOUGH IMPLEMENTATION PARAMETERS
  unsigned int orientation_accumulators_num = 30;
  unsigned int gaussian_sphere_points_num = 450;
  float accumulator_peak_threshold = 0.8;
  float min_radius = 0.25;
  float max_radius = 0.35;
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

  // Gaussian Sphere Top Biased
  std::vector<double> weights_biased;
  std::vector<Eigen::Matrix<double, 3, 1> > means_biased;
  std::vector<Eigen::Matrix<double, 3, 1> > std_devs_biased;
  weights_biased.push_back(1.0);
  Eigen::Matrix<double, 3, 1> mean_eigen_biased(0, 0, 1.0);
  means_biased.push_back(mean_eigen_biased);
  Eigen::Matrix<double, 3, 1> std_dev_eigen_biased(0.5, 0.5, 0.5);
  std_devs_biased.push_back(std_dev_eigen_biased);

  GaussianMixtureModel gmm_biased(weights_biased, means_biased,
                                  std_devs_biased);
  GaussianSphere gaussian_sphere_biased(gmm_biased, gaussian_sphere_points_num,
                                        orientation_accumulators_num);

  // Gaussian Sphere Strong Top Biased
  std::vector<double> weights_super_biased;
  std::vector<Eigen::Matrix<double, 3, 1> > means_super_biased;
  std::vector<Eigen::Matrix<double, 3, 1> > std_devs_super_biased;
  weights_super_biased.push_back(1.0);
  Eigen::Matrix<double, 3, 1> mean_eigen_super_biased(0, 0, 1.0);
  means_super_biased.push_back(mean_eigen_super_biased);
  Eigen::Matrix<double, 3, 1> std_dev_eigen_super_biased(0.05, 0.05, 0.05);
  std_devs_super_biased.push_back(std_dev_eigen_super_biased);

  GaussianMixtureModel gmm_super_biased(
      weights_super_biased, means_super_biased, std_devs_super_biased);
  GaussianSphere gaussian_sphere_super_biased(gmm_super_biased,
                                              gaussian_sphere_points_num,
                                              orientation_accumulators_num);

  // // HOUGH HYBRID SUPER BIASED (soft-voting)  
  boost::shared_ptr<CylinderFittingHough> cylinder_fitting(
      new CylinderFittingHough(
          gaussian_sphere_super_biased, (unsigned int)angle_bins,
          (unsigned int)radius_bins, (unsigned int)position_bins,
          (float)min_radius, (float)max_radius,
          (float)accumulator_peak_threshold, CylinderFittingHough::HYBRID,
          false, true));

  pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne_org;
  ne_org.setNormalEstimationMethod(ne_org.AVERAGE_3D_GRADIENT);
  ne_org.setMaxDepthChangeFactor(0.02f);
  ne_org.setNormalSmoothingSize(10.0f);

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

  // Correct the orientation of the normals. 
  // To make the normals point out the camera. 
  for (unsigned int i = 0; i < cloud_normals->size(); ++i) {
    double dot_product =
        cloud_normals->points[i].getNormalVector3fMap().dot(Eigen::Vector3f(
            point_cloud->points[i].x, point_cloud->points[i].y,
            point_cloud->points[i].z));
    if (dot_product < 0.0) {
      cloud_normals->at(i).normal_x = -cloud_normals->at(i).normal_x;
      cloud_normals->at(i).normal_y = -cloud_normals->at(i).normal_y;
      cloud_normals->at(i).normal_z = -cloud_normals->at(i).normal_z;
    }
  }

  FittingData fitting_data = cylinder_fitting->fit(point_cloud, cloud_normals);

  auto fitting_result = fitting_data.parameters;
  // 
  // 参数顺序：[c1, c2, c3, d1, d2, d3, r, height]
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
