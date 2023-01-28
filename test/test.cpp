#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <unordered_set>
#include <vector>

#include <opencv2/opencv.hpp>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>

#include "helper.h"
#include "tasks.hpp"

cv::Mat g_canvas(500, 1900, CV_8UC3, cv::Vec3b(255U, 255U, 255U));
#define DEBUG_IMG

class DataProvider {
public:
  DataProvider(const std::string &dir) {
    const std::filesystem::path dir_data{dir};
    std::unordered_set<std::string> all_files;
    for (auto const &dir_entry :
         std::filesystem::directory_iterator{dir_data}) {
      all_files.insert(dir_entry.path());
    }

    for (int i = 0; i < 100000; ++i) {
      const std::string base = dir + std::to_string(i);
      const std::string a = base + ".pcd";
      const std::string b = base + ".txt";
      if (all_files.find(a) != all_files.end() and
          all_files.find(b) != all_files.end()) {
        todos_.push({a, b});
      } else {
        break;
      }
    }
  }

  bool fine() { return !todos_.empty(); }

  std::pair<std::string, std::string> then() {
    auto ret = todos_.front();
    todos_.pop();
    return ret;
  }

  static std::pair<PointCloudT::Ptr, Pose>
  load(const std::pair<std::string, std::string> &in) {
    PointCloudT::Ptr cloud(new PointCloudT);
    if (pcl::io::loadPCDFile(in.first, *cloud) == -1) //* load the file
    {
      PCL_ERROR("Couldn't read \n");
      cloud = nullptr;
    }

    Pose pose;
    std::ifstream is(in.second);
    is >> pose.position.x >> pose.position.y >> pose.position.z;
    return {cloud, pose};
  }

private:
  std::queue<std::pair<std::string, std::string>> todos_;
};

std::vector<float> get_max(PointCloudT::Ptr pc) {
  float min_z = pc->points[0].z;
  std::vector<float> ret{pc->points[0].x, pc->points[0].x, pc->points[0].y,
                         pc->points[0].y};

  for (const auto &pt : pc->points) {
    ret[0] = std::max(ret[0], pt.x);
    ret[1] = std::min(ret[1], pt.x);
    ret[2] = std::max(ret[2], pt.y);
    ret[3] = std::min(ret[3], pt.y);
    min_z = std::min(min_z, pt.z);
  }

  return ret;
}

void draw(cv::Mat &canvas, const std::vector<float> &lim,
          const PointCloudT::Ptr &pc, const cv::Vec3b color) {
  for (const auto &pt : pc->points) {
    const float x = (pt.x - lim[1]) / (lim[0] - lim[1]);
    const float y = (pt.y - lim[3]) / (lim[2] - lim[3]);
    const int row = (int)(y * canvas.rows);
    const int col = (int)(x * canvas.cols);
    if (row >= 0 && row < canvas.rows && col >= 0 && col < canvas.cols) {
      canvas.at<cv::Vec3b>((int)(y * canvas.rows), (int)(x * canvas.cols)) =
          color;
    }
  }
}

PointCloudT::Ptr filter(const PointCloudT::Ptr &pc) {
  PointCloudT::Ptr ret(new PointCloudT);
  for (const auto &pt : pc->points) {
    if (pt.z > 0.0F) {
      ret->points.push_back(pt);
      ret->points.back().z = 0.0F;
    }
  }
  return ret;
}

class Localizer {
public:
  Localizer() = delete;
  Localizer(const PointCloudT::Ptr map) { set_map(map); }

  static PointCloudT::Ptr filter_ground(const PointCloudT::Ptr pc_in) {
    PointCloudT::Ptr ret(new PointCloudT);
    for (const auto &pt : pc_in->points) {
      if (pt.z > 0.0F) {
        ret->points.push_back(pt);
        ret->points.back().z = 0.0F;
      }
    }
    return ret;
  }

  static PointCloudT::Ptr volxelize(const PointCloudT::Ptr pc_in,
                                    const float voxel_size) {
    PointCloudT::Ptr ret(new PointCloudT);
    pcl::VoxelGrid<PointT> sor;
    sor.setInputCloud(pc_in);
    sor.setLeafSize(voxel_size, voxel_size, voxel_size);
    sor.filter(*ret);
    return ret;
  }

  void set_map(const PointCloudT::Ptr pc_in) {
    map_ = volxelize(filter_ground(pc_in), 0.1F);
  }

  void run(const PointCloudT::Ptr pc_in) {
    assert(map_ != nullptr);
    PointCloudT::Ptr scan_lite = volxelize(filter_ground(pc_in), 0.4F);
    PointCloudT::Ptr transformed(new PointCloudT);
    pcl::transformPointCloud(*scan_lite, *transformed,
                             get_assumped_transform(total_transform_));

#ifdef DEBUG_IMG
    const auto lim = get_max(map_);
    draw(g_canvas, lim, map_, cv::Vec3b(192U, 192U, 192U));
    draw(g_canvas, lim, scan_lite, cv::Vec3b(255U, 0U, 0U));
    draw(g_canvas, lim, transformed, cv::Vec3b(0U, 0U, 255U));
#endif // DEBUG_IMG

    const Eigen::Matrix4f delta = align(transformed);

    total_transform_ = delta * total_transform_;
    add_transform(total_transform_);

#ifdef DEBUG_IMG
    cv::imshow("canvas", g_canvas);
    cv::waitKey(100);
#endif // DEBUG_IMG
  }

  Eigen::Matrix4f get() const { return total_transform_; }

  float check(const float x, const float y, const float z) const {
    const Eigen::Vector3f t(x, y, z);
    const Eigen::Vector3f r = get().block(0, 3, 3, 1);
    return (t - r).norm();
  }

private:
  PointCloudT::Ptr map_ = nullptr;
  std::deque<Eigen::Vector3f> positions_;
  static constexpr size_t pos_buffer_size_ = 5U;
  Eigen::Matrix4f total_transform_ = Eigen::Matrix4f::Identity();

  Eigen::Matrix4f align(const PointCloudT::Ptr pc_in) {
    PointCloudT::Ptr cloud_icp(new PointCloudT);
    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setInputSource(pc_in);
    icp.setInputTarget(map_);
    icp.setMaximumIterations(50);
    icp.setUseReciprocalCorrespondences(true);
    icp.align(*cloud_icp);
    return icp.getFinalTransformation();
  }

  Eigen::Matrix4f
  get_assumped_transform(const Eigen::Matrix4f &last_transform) {
    Eigen::Matrix4f delta_transform = Eigen::Matrix4f::Identity();
    if (positions_.size() >= pos_buffer_size_) {
      delta_transform.block(0, 3, 3, 1) =
          (positions_.back() - positions_.front()) /
          (float)(positions_.size() - 1U);
    }
    return delta_transform * last_transform;
  }

  void add_transform(const Eigen::Matrix4f &last_transform) {
    positions_.push_back(last_transform.block(0, 3, 3, 1));
    while (positions_.size() > pos_buffer_size_) {
      positions_.pop_front();
    }
  }
};

int main() {
  char const *home = getenv("HOME");
  const std::string prefix{std::string{home} +
                           "/workspace/udacity/localization/data2/"};
  const std::string path_map{
      std::string{home} + "/workspace/udacity/localization/c3-project/map.pcd"};
  DataProvider warehouse(prefix);
  PointCloudT::Ptr map(new PointCloudT);
  PointCloudT::Ptr map_lite(new PointCloudT);
  if (pcl::io::loadPCDFile(path_map, *map) == -1) //* load the file
  {
    PCL_ERROR("Couldn't read map \n");
    map = nullptr;
    return EXIT_FAILURE;
  }
  const auto lim = get_max(map);

  // map = filter(map);
  // volxelize(map, map_lite, 0.1F);

  // Eigen::Matrix4f total_transform = Eigen::Matrix4f::Identity();
  // Eigen::Matrix4f delta_transform = Eigen::Matrix4f::Identity();
  // std::deque<Eigen::Vector3f> positions;

  Localizer loc(map);
  float max_error = 0.0F;

  while (warehouse.fine()) {
    g_canvas.setTo(cv::Vec3b(255U, 255U, 255U));
    auto data = DataProvider::load(warehouse.then());
    std::cout << data.first->size() << std::endl
              << data.second.position.x << " " << data.second.position.y << " "
              << data.second.position.z << std::endl;

    loc.run(data.first);
    auto gt = data.second.position;
    const float error = loc.check(gt.x, gt.y, gt.z);
    max_error = std::max(error, max_error);
    std::cout << error << std::endl;

    // // from here it should be the same as in project
    // PointCloudT::Ptr scan = filter(data.first);
    // PointCloudT::Ptr scan_lite(new PointCloudT);
    // PointCloudT::Ptr map_lite_inv(new PointCloudT);
    // PointCloudT::Ptr transformed(new PointCloudT);
    // volxelize(scan, scan_lite, 0.4F);
    // if (positions.size() >= 3) {
    //   delta_transform = Eigen::Matrix4f::Identity();
    //   delta_transform.block(0, 3, 3, 1) =
    //       (positions.back() - positions.front()) /
    //       (float)(positions.size() - 1U);
    //   // std::cout << delta_transform << std::endl;
    // } else {
    //   delta_transform = Eigen::Matrix4f::Identity();
    // }
    // pcl::transformPointCloud(*scan_lite, *transformed,
    //                          delta_transform * total_transform);
    // // pcl::transformPointCloud(*map_lite, *map_lite_inv, (delta_transform *
    // // total_transform).inverse());

    // const auto ret = alignICP(transformed, map_lite, 50);
    // const Eigen::Matrix4f transform = ret.first;

    // total_transform = transform * total_transform;
    // positions.push_back(total_transform.block(0, 3, 3, 1));
    // while (positions.size() > 5) {
    //   positions.pop_front();
    // }
    // std::cout << total_transform(0, 3) << " " << total_transform(1, 3) << " "
    //           << total_transform(2, 3) << std::endl;

    // cv::Mat canvas(500, 1900, CV_8UC3, cv::Vec3b(255U, 255U, 255U));
    // // draw(canvas, lim, map, cv::Vec3b(128U,128,128U));
    // draw(canvas, lim, map_lite, cv::Vec3b(255U, 128U, 128U));
    // // draw(canvas, lim, map_lite_inv, cv::Vec3b(0U,0U,0U));
    // draw(canvas, lim, scan_lite, cv::Vec3b(128U, 255U, 128U));
    // // draw(canvas, lim, transformed, cv::Vec3b(0U,0U,0U));
    // pcl::transformPointCloud(*scan_lite, *transformed, total_transform);
    // draw(canvas, lim, transformed, cv::Vec3b(128U, 128U, 255U));
    // // std::cout << (int)ret.second << std::endl;

    // cv::imshow("canvas", canvas);
    // cv::waitKey(100);
  }
  std::cout << max_error << std::endl;
}