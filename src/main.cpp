#include <pcl/point_cloud.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <vector>

std::vector<int> myRansacLine(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
                              const double &dist_threshold,
                              const double &target_probability,
                              const double &inlier_probability)
{
  unsigned int seed = time(NULL);
  const double max_iterations = std::log(1 - target_probability) / std::log(1 - inlier_probability * inlier_probability);
  std::cout << "estimated max_iterations " << max_iterations << std::endl;

  std::vector<int> best_inliers;
  for (unsigned int i = 0; i < max_iterations; ++i)
  {
    const size_t index1 = rand_r(&seed) % cloud->points.size();
    const size_t index2 = rand_r(&seed) % cloud->points.size();
    if (index1 == index2)
    {
      continue;
    }

    const Eigen::Vector3d point1(cloud->points[index1].x, cloud->points[index1].y, cloud->points[index1].z);
    const Eigen::Vector3d point2(cloud->points[index2].x, cloud->points[index2].y, cloud->points[index2].z);

    const Eigen::Vector3d line_vector = point2 - point1;
    std::vector<int> inliers;
    for (size_t j = 0; j < cloud->points.size(); ++j)
    {
      const Eigen::Vector3d point0(cloud->points[j].x, cloud->points[j].y, cloud->points[j].z);
      const double distance = (line_vector.cross(point1 - point0)).norm() / line_vector.norm();
      if (distance <= dist_threshold)
      {
        inliers.push_back(j);
      }
    }
    if (inliers.size() > best_inliers.size())
    {
      best_inliers = inliers;
    }
  }

  return best_inliers;
}

int main()
{
  unsigned int seed = time(NULL);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud->width = 500;
  cloud->height = 1;  // unorganized point cloud dataset
  cloud->is_dense = true;
  cloud->points.resize(cloud->width * cloud->height);

  // generate data points
  for (size_t i = 0; i < cloud->points.size (); ++i)
  {
    cloud->points[i].x = 1024 * rand_r(&seed) / (RAND_MAX + 1.0);
    cloud->points[i].y = 1 - cloud->points[i].x;
    if (i % 2 == 0)
    {
      cloud->points[i].z = 0;  // inliers
    }
    else
    {
      cloud->points[i].z = 1024 * rand_r(&seed) / (RAND_MAX + 1.0);  // outliers
    }
  }

  // PCL RANSAC line fitting
  std::vector<int> inliers;
  pcl::SampleConsensusModelLine<pcl::PointXYZ>::Ptr model_l(new pcl::SampleConsensusModelLine<pcl::PointXYZ> (cloud));
  pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_l);
  ransac.setDistanceThreshold(0.01);
  ransac.computeModel();
  ransac.getInliers(inliers);
  std::cout << "pcl inliers.size() " << inliers.size() << std::endl;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_line_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::copyPointCloud(*cloud, inliers, *pcl_line_cloud);
  for (pcl::PointXYZRGB &p : pcl_line_cloud->points)
  {
    p.r = 255;
  }

  // My RANSAC line fitting
  inliers = myRansacLine(cloud, 0.01, 0.99, 0.1);
  std::cout << "my inliers.size() " << inliers.size() << std::endl;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr my_line_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::copyPointCloud(*cloud, inliers, *my_line_cloud);
  for (pcl::PointXYZRGB &p : my_line_cloud->points)
  {
    p.b = 255;
  }

  // Visualization
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor(0, 0, 0);
  viewer->addPointCloud(cloud, "sample cloud");
  //  viewer->addPointCloud(pcl_line_cloud, "pcl line cloud");
  viewer->addPointCloud(my_line_cloud, "my line cloud");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  //  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "pcl line cloud");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "my line cloud");
  viewer->initCameraParameters();

  while (!viewer->wasStopped())
  {
    viewer->spinOnce(100);
    boost::this_thread::sleep(boost::posix_time::microseconds(100000));
  }
}
