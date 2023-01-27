#include "tasks.hpp"

#include <iostream>

#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>

void volxelize(typename pcl::PointCloud<PointT>::Ptr scanCloud,
               typename pcl::PointCloud<PointT>::Ptr cloudFiltered,
               const float voxel_size = 0.2F) {
  pcl::VoxelGrid<PointT> sor;
  sor.setInputCloud(scanCloud);
  sor.setLeafSize(voxel_size, voxel_size, voxel_size);
  sor.filter(*cloudFiltered);
}

std::pair<Eigen::Matrix4f, bool>
alignICP(typename pcl::PointCloud<PointT>::Ptr pc_from,
         typename pcl::PointCloud<PointT>::Ptr pc_to,
         const int max_interation) {
  PointCloudT::Ptr cloud_icp(new PointCloudT);
  pcl::IterativeClosestPoint<PointT, PointT> icp;
  icp.setInputSource(pc_from);
  icp.setInputTarget(pc_to);
  icp.setMaximumIterations(max_interation);
  // icp.setTransformationEpsilon(1e-6);
  icp.setUseReciprocalCorrespondences(true);
  icp.align(*cloud_icp);
  return {icp.getFinalTransformation(), icp.hasConverged()};
}
