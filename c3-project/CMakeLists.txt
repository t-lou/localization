project(Localization)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_minimum_required (VERSION 3.5)

# -g allows for gdb debugging
# turn on -03 for best performance
add_definitions(-std=c++14 -O3)

set(CXX_FLAGS "-pthread -fPIC -DNDEBUG -Wextra")
#set(CMAKE_CXX_FLAGS "${CXX_FLAGS} -lboost_filesystem -lboost_system -lboost_numpy36 -lboost_program_options -lboost_python36 -lboost_system")

set(sources c3-main.cpp helper.cpp tasks.cpp)

set(CARLALIB_DIR ${CMAKE_SOURCE_DIR}/libcarla-install/lib)
set(carla_libs ${CARLALIB_DIR}/libcarla_client.a ${CARLALIB_DIR}/librpc.a ${CARLALIB_DIR}/libDebugUtils.a ${CARLALIB_DIR}/libDetour.a ${CARLALIB_DIR}/libDetourCrowd.a ${CARLALIB_DIR}/libDetourTileCache.a ${CARLALIB_DIR}/libRecast.a)

link_directories(libcarla-install/lib)
include_directories(libcarla-install/include)
include_directories(libcarla-install/include/system)
include_directories(rpclib/include)

find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

add_executable(cloud_loc ${sources} )

target_link_libraries(cloud_loc
    ${PCL_LIBRARIES}
    ${carla_libs}
    boost_filesystem
    boost_system
    boost_numpy36
    boost_program_options
    boost_python36
    boost_system
    pthread
)
