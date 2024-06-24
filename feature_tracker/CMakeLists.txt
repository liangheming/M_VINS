cmake_minimum_required(VERSION 3.0.2)
project(feature_tracker)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
  cv_bridge
  camera_model
  geometry_msgs
)

find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)


catkin_package()
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)
set(SRC_LIST src/feature_tracker.cpp)
add_executable(${PROJECT_NAME}_node src/feature_tracker_node.cpp ${SRC_LIST})

target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}  ${OpenCV_LIBS} yaml-cpp
)

install(TARGETS ${PROJECT_NAME}_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES
  launch/feature_tracker.launch
  config/camera.yaml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)