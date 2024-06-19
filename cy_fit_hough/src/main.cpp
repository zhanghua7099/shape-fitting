#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>
#include <vector>

#include "cylinder_fitting_hough.h"
#include "cylinder_fitting_ransac.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

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


int add(int i, int j) {
    return i + j;
}

Eigen::MatrixXd inv(const Eigen::MatrixXd &xs)
{
//   SphericalGrid spherical_grid(10, 10);
  return xs.inverse();
}

Eigen::VectorXd cylinder_fitting_Figueiredo(const Eigen::MatrixXd &points)
{
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


    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (int i = 0; i < points.rows(); ++i) {
        pcl::PointXYZ point;
        point.x = points(i, 0);
        point.y = points(i, 1);
        point.z = points(i, 2);
        point_cloud->points.push_back(point);
    }
    point_cloud->width = point_cloud->points.size();
    point_cloud->height = 1;
    point_cloud->is_dense = true;

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

    // 参数顺序：[c1, c2, c3, d1, d2, d3, r, height]

    Eigen::VectorXd cy_params(7);
    cy_params(0) = fitting_result(0);
    cy_params(1) = fitting_result(1);
    cy_params(2) = fitting_result(2);
    cy_params(3) = fitting_result(3);
    cy_params(4) = fitting_result(4);
    cy_params(5) = fitting_result(5);
    cy_params(6) = fitting_result(6);
    return cy_params;
}


PYBIND11_MODULE(_core, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cy_fit_hough

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("cylinder_fitting_Figueiredo", &cylinder_fitting_Figueiredo, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("inv", &inv, R"pbdoc(
        Inverse the matrix
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
