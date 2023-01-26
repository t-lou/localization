#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <queue>
#include <string>
#include <unordered_set>

#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <opencv2/opencv.hpp>

#include "helper.h"
#include "tasks.hpp"

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

std::vector<float> get_max(PointCloudT::Ptr pc)
{
  float min_z = pc->points[0].z;
  std::vector<float> ret{pc->points[0].x, pc->points[0].x, pc->points[0].y, pc->points[0].y};

  for (const auto& pt : pc->points)
  {
    ret[0] = std::max(ret[0], pt.x);
    ret[1] = std::min(ret[1], pt.x);
    ret[2] = std::max(ret[2], pt.y);
    ret[3] = std::min(ret[3], pt.y);
    min_z = std::min(min_z, pt.z);
  }

  std::cout << ret[0] << " " << ret[1] << " " << ret[2] << " " << ret[3] << std::endl;
  std::cout << min_z << std::endl;

  return ret;
}

void draw(cv::Mat& canvas, const std::vector<float>& lim, const PointCloudT::Ptr& pc, const cv::Vec3b color)
{
  for (const auto& pt : pc->points)
  {
    const float x = (pt.x - lim[1]) / (lim[0] - lim[1]);
    const float y = (pt.y - lim[3]) / (lim[2] - lim[3]);
    const int row = (int)(y * canvas.rows);
    const int col = (int)(x * canvas.cols);
    if (row >= 0 && row < canvas.rows && col >= 0 && col < canvas.cols)
    {
      canvas.at<cv::Vec3b>((int)(y * canvas.rows), (int)(x * canvas.cols)) = color;
    }
  }
}

PointCloudT::Ptr filter(const PointCloudT::Ptr& pc)
{
  PointCloudT::Ptr ret(new PointCloudT);
  for (const auto& pt : pc->points)
  {
    if (pt.z > 0.0F)
    {
      ret->points.push_back(pt);
    }
  }
  return ret;
}

int main() { 
  char const* home = getenv("HOME");
  const std::string prefix{std::string{home} + "/workspace/udacity/localization/data2/"};
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

  map = filter(map);
  volxelize(map, map_lite);

  Eigen::Matrix4f total_transform = Eigen::Matrix4f::Identity();

  while (warehouse.fine()) {
    auto data = DataProvider::load(warehouse.then());
    std::cout << data.first->size() << std::endl
              << data.second.position.x << " " << data.second.position.y << " "
              << data.second.position.z << std::endl;

    // from here it should be the same as in project
    PointCloudT::Ptr scan = filter(data.first);
    PointCloudT::Ptr scan_lite(new PointCloudT);
    PointCloudT::Ptr transformed(new PointCloudT);
    volxelize(scan, scan_lite);
    pcl::transformPointCloud(*scan_lite, *transformed, total_transform);

    const auto ret = alignICP(transformed, map_lite, 50);
    const Eigen::Matrix4f transform = ret.first;

    total_transform = transform * total_transform;
    std::cout << total_transform(0, 3) << " " << total_transform(1, 3) << " "
              << total_transform(2, 3) << std::endl;

    pcl::transformPointCloud(*transformed, *transformed, transform);

    cv::Mat canvas(500, 1900, CV_8UC3, cv::Vec3b(255U,255U,255U));

    draw(canvas, lim, map, cv::Vec3b(128U,128,128U));
    draw(canvas, lim, map_lite, cv::Vec3b(255U,128U,128U));
    draw(canvas, lim, scan_lite, cv::Vec3b(128U,255U,128U));
    draw(canvas, lim, transformed, cv::Vec3b(128U,128U,255U));

    cv::imshow("canvas", canvas);
    cv::waitKey(100);
  }
}