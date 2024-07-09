cmake_minimum_required(VERSION 3.0.2)
project(vins_estimator)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  roscpp
  std_msgs
  nav_msgs
  tf
)

add_compile_definitions(SOPHUS_USE_BASIC_LOGGING)

find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES vins_estimator
#  CATKIN_DEPENDS cv_bridge geometry_msgs roscpp std_msgs tf
#  DEPENDS system_lib
)


include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Ceres_INCLUDE_DIRS}
)

set(SRC_LIST src/estimator/commons.cpp
             src/estimator/integration.cpp
             src/estimator/initial_sfm.cpp
             src/estimator/initial_alignment.cpp
             src/estimator/feature_manager.cpp
             src/estimator/sw_estimator.cpp
             src/estimator/factors/imu_factor.cpp
             src/estimator/factors/projection_factor.cpp
             src/estimator/factors/pose_parameterization.cpp
             src/estimator/factors/marginalization_factor.cpp
            )

add_executable(${PROJECT_NAME}_node src/vins_estimator_node.cpp ${SRC_LIST})


target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  yaml-cpp
)

install(TARGETS ${PROJECT_NAME}_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES
  launch/vins_estimator.launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
