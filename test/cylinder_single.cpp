/*
Copyright 2018 Rui Miguel Horta Pimentel de Figueiredo

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

/*!    
    \author Rui Figueiredo : ruipimentelfigueiredo
*/

#include <ctime>
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH
#include "cylinder_fitting_hough.h"
#include "cylinder_fitting_ransac.h"
#include "helpers.h"
#include <chrono>
using namespace std;
using namespace std::chrono;

int main (int argc, char** argv)
{
	// HOUGH PARAMETERS
	unsigned int angle_bins=30;
	unsigned int radius_bins=10;
	unsigned int position_bins=100;

	// OUR HOUGH IMPLEMENTATION PARAMETERS
	unsigned int orientation_accumulators_num=30;
	unsigned int gaussian_sphere_points_num=450;
	float accumulator_peak_threshold=0.8;
	float min_radius=0.25;
	float max_radius=0.35;
	
	// RANSAC PARAMETERS
	float normal_distance_weight=0.5;
	float distance_threshold=0.05;
	bool do_refine=false;
	bool visualize = true;

	boost::shared_ptr<VisualizeFittingData> visualizer;
	
	if(visualize)
		visualizer=boost::shared_ptr<VisualizeFittingData>(new VisualizeFittingData());

	// Voting Sphere Uniform Grid
	SphericalGrid spherical_grid(gaussian_sphere_points_num,orientation_accumulators_num);

	// Gaussian Sphere Uniform
	std::vector<double> weights;
	std::vector<Eigen::Matrix<double, 3 ,1> > means;
	std::vector<Eigen::Matrix<double, 3 ,1> > std_devs;
	weights.push_back(1.0);
	Eigen::Matrix<double, 3 ,1> mean_eigen(0,0,0);
	means.push_back(mean_eigen);
	Eigen::Matrix<double, 3 ,1> std_dev_eigen(0.5,0.5,0.5);
	std_devs.push_back(std_dev_eigen);

	GaussianMixtureModel gmm(weights,means,std_devs);
	GaussianSphere gaussian_sphere(gmm,gaussian_sphere_points_num,orientation_accumulators_num);

	// Gaussian Sphere Top Biased
	std::vector<double> weights_biased;
	std::vector<Eigen::Matrix<double, 3 ,1> > means_biased;
	std::vector<Eigen::Matrix<double, 3 ,1> > std_devs_biased;
	weights_biased.push_back(1.0);
	Eigen::Matrix<double, 3 ,1> mean_eigen_biased(0,0,1.0);
	means_biased.push_back(mean_eigen_biased);
	Eigen::Matrix<double, 3 ,1> std_dev_eigen_biased(0.5,0.5,0.5);
	std_devs_biased.push_back(std_dev_eigen_biased);

	GaussianMixtureModel gmm_biased(weights_biased,means_biased,std_devs_biased);
	GaussianSphere gaussian_sphere_biased(gmm_biased,gaussian_sphere_points_num,orientation_accumulators_num);

	// Gaussian Sphere Strong Top Biased
	std::vector<double> weights_super_biased;
	std::vector<Eigen::Matrix<double, 3 ,1> > means_super_biased;
	std::vector<Eigen::Matrix<double, 3 ,1> > std_devs_super_biased;
	weights_super_biased.push_back(1.0);
	Eigen::Matrix<double, 3 ,1> mean_eigen_super_biased(0,0,1.0);
	means_super_biased.push_back(mean_eigen_super_biased);
	Eigen::Matrix<double, 3 ,1> std_dev_eigen_super_biased(0.05,0.05,0.05);
	std_devs_super_biased.push_back(std_dev_eigen_super_biased);

	GaussianMixtureModel gmm_super_biased(weights_super_biased,means_super_biased,std_devs_super_biased);
	GaussianSphere gaussian_sphere_super_biased(gmm_super_biased,gaussian_sphere_points_num,orientation_accumulators_num);

	std::vector<boost::shared_ptr<CylinderFitting> > cylinder_segmentators;

	// HOUGH RABANI
	cylinder_segmentators.push_back(boost::shared_ptr<CylinderFittingHough> (new CylinderFittingHough(gaussian_sphere,(unsigned int)angle_bins,(unsigned int)radius_bins,(unsigned int)position_bins,(float)min_radius, (float)max_radius,(float)accumulator_peak_threshold,CylinderFittingHough::NORMAL, false, false)));

	// HOUGH HYBRID (soft-voting)
	cylinder_segmentators.push_back(boost::shared_ptr<CylinderFittingHough> (new CylinderFittingHough(gaussian_sphere,(unsigned int)angle_bins,(unsigned int)radius_bins,(unsigned int)position_bins,(float)min_radius, (float)max_radius,(float)accumulator_peak_threshold,CylinderFittingHough::HYBRID,false, true)));

	// HOUGH HYBRID BIASED (soft-voting)
	cylinder_segmentators.push_back(boost::shared_ptr<CylinderFittingHough> (new CylinderFittingHough(gaussian_sphere_biased,(unsigned int)angle_bins,(unsigned int)radius_bins,(unsigned int)position_bins,(float)min_radius, (float)max_radius,(float)accumulator_peak_threshold,CylinderFittingHough::HYBRID,false, true)));	

	// HOUGH HYBRID SUPER BIASED (soft-voting)
	cylinder_segmentators.push_back(boost::shared_ptr<CylinderFittingHough> (new CylinderFittingHough(gaussian_sphere_super_biased,(unsigned int)angle_bins,(unsigned int)radius_bins,(unsigned int)position_bins,(float)min_radius, (float)max_radius,(float)accumulator_peak_threshold,CylinderFittingHough::HYBRID,false, true)));
	
	// HOUGH HYBRID (soft-voting ICARSC)
	cylinder_segmentators.push_back(boost::shared_ptr<CylinderFittingHough> (new CylinderFittingHough(gaussian_sphere,(unsigned int)angle_bins,(unsigned int)radius_bins,(unsigned int)position_bins,(float)min_radius, (float)max_radius,(float)accumulator_peak_threshold,2,false, true)));













	std::string detections_frame_id="world";
	std::string marker_detections_namespace_="detections";

	// Segment cylinder and store results
	std::vector<Eigen::VectorXf > detections;
	std::vector<float> position_errors;

	pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne_org;
	ne_org.setNormalEstimationMethod (ne_org.AVERAGE_3D_GRADIENT);
	ne_org.setMaxDepthChangeFactor(0.02f);
	ne_org.setNormalSmoothingSize(10.0f);

	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
	ne.setKSearch (6);
	ne.setSearchMethod (tree);

	pcl::PointCloud<PointT>::Ptr point_cloud(new pcl::PointCloud<PointT>());
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>());

    // read pcl
    if (pcl::io::loadPCDFile<PointT> ("/home/zhy/CppProject/shape-fitting/dataset/cylinder_occlusion/point_clouds/cloud_000012_noise_000000.pcd", *point_cloud))
    {
        pcl::console::print_error ("Error loading cloud file!\n");
        return (1);
    }

    // compute normal
    if(point_cloud->isOrganized())
    {
        ne_org.setInputCloud (point_cloud);
        ne_org.compute (*cloud_normals);
    }
    else
    {
        ne.setInputCloud (point_cloud);
        ne.compute (*cloud_normals);
    }
    for(unsigned int i=0; i<cloud_normals->size();++i)
    {
        double dot_product=cloud_normals->points[i].getNormalVector3fMap ().dot(Eigen::Vector3f(point_cloud->points[i].x,point_cloud->points[i].y,point_cloud->points[i].z));
        if(dot_product<0.0)
        {
            cloud_normals->at(i).normal_x=-cloud_normals->at(i).normal_x;
            cloud_normals->at(i).normal_y=-cloud_normals->at(i).normal_y;
            cloud_normals->at(i).normal_z=-cloud_normals->at(i).normal_z;
        }
    }

	for (unsigned int d=0; d < cylinder_segmentators.size();++d)
	{
        /* FIT */



        FittingData model_params=cylinder_segmentators[d]->fit(point_cloud, cloud_normals);

        /* END FIT */

        /* VISUALIZE */
        if(visualize)
        {
            auto viewer = model_params.visualize(point_cloud, visualizer);
            while (!viewer->wasStopped ())
            {
                viewer->spinOnce (100);
            }
        }
        /* END VISUALIZE */
	}

	return (0);
}
