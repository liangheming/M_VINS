cmake_minimum_required(VERSION 3.8)
project(vins_estimator)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_definitions(-DSOPHUS_USE_BASIC_LOGGING)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-O3)
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(yaml-cpp REQUIRED)

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${Ceres_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
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
ament_target_dependencies(${PROJECT_NAME}_node rclcpp std_msgs nav_msgs sensor_msgs geometry_msgs)
target_link_libraries(${PROJECT_NAME}_node
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  yaml-cpp
)

install(TARGETS ${PROJECT_NAME}_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})


ament_package()
