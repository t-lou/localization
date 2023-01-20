#include "tasks.hpp"

#include <iostream>

#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/voxel_grid.h>

void volxelize(typename pcl::PointCloud<PointT>::Ptr scanCloud,
               typename pcl::PointCloud<PointT>::Ptr cloudFiltered)
{
    pcl::VoxelGrid<PointT> sor;
    constexpr float voxel_size = 0.1f;
    sor.setInputCloud (scanCloud);
    sor.setLeafSize (voxel_size, voxel_size, voxel_size);
    sor.filter (*cloudFiltered);
}

std::pair<Eigen::Matrix4f, bool> alignNDT(
    typename pcl::PointCloud<PointT>::Ptr pc_from,
    typename pcl::PointCloud<PointT>::Ptr pc_to,
    const int max_interation)
{
	PointCloudT::Ptr cloud_icp (new PointCloudT);
	pcl::IterativeClosestPoint<PointT, PointT> icp;
	icp.setInputSource (pc_from);
	icp.setInputTarget (pc_to);
	icp.setMaximumIterations (max_interation);
	icp.align (*cloud_icp);
	return {icp.getFinalTransformation (), icp.hasConverged ()};
}

std::pair<Eigen::Matrix4f, bool> alignICP(
    typename pcl::PointCloud<PointT>::Ptr pc_from,
    typename pcl::PointCloud<PointT>::Ptr pc_to,
    const int max_interation)
{
    pcl::NormalDistributionsTransform<PointT, PointT> ndt;
    ndt.setTransformationEpsilon (0.01);
    ndt.setStepSize (0.1);
    ndt.setResolution (1.0);
    ndt.setMaximumIterations (max_interation);
    ndt.setInputSource (pc_from);
    ndt.setInputTarget (pc_to);
    ndt.hasConverged ();
    return {ndt.getFinalTransformation (), ndt.hasConverged ()};
}