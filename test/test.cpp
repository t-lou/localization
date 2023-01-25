#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <unordered_set>

#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

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

int main() { 
  char const* home = getenv("HOME");
  const std::string prefix{std::string{home} + "/workspace/udacity/localization/data1/"};
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
  volxelize(map, map_lite);
  while (warehouse.fine()) {
    auto data = DataProvider::load(warehouse.then());
    std::cout << data.first->size() << std::endl
              << data.second.position.x << " " << data.second.position.y << " "
              << data.second.position.z << std::endl;

    // from here it should be the same as in project
    PointCloudT::Ptr scan = data.first;
    PointCloudT::Ptr scan_lite(new PointCloudT);
    volxelize(scan, scan_lite);

    const auto ret = alignICP(scan_lite, map_lite, 20);
    const Eigen::Matrix4f transform = ret.first;
    std::cout << (ret.second ? "converged" : "unconverged") << std::endl;
    std::cout << transform(0, 3) << " " << transform(1, 3) << " "
              << transform(2, 3) << std::endl;
    // pose = getPose(totalTransform);
  }
}