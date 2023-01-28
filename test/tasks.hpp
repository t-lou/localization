#pragma once

#include <memory>

#include <pcl/common/projection_matrix.h>

#include "helper.h"

void volxelize(typename pcl::PointCloud<PointT>::Ptr scanCloud,
               typename pcl::PointCloud<PointT>::Ptr cloudFiltered,
               const float voxel_size);

std::pair<Eigen::Matrix4f, bool>
alignICP(typename pcl::PointCloud<PointT>::Ptr pc_from,
         typename pcl::PointCloud<PointT>::Ptr pc_to, int max_interation);