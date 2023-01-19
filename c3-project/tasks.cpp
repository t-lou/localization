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