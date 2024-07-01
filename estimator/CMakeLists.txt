cmake_minimum_required(VERSION 3.0.2)
project(estimator)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

add_compile_definitions(SOPHUS_USE_BASIC_LOGGING)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  nav_msgs
  roscpp
  std_msgs
  tf
)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)


## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES estimator
#  CATKIN_DEPENDS cv_bridge geometry_msgs nav_msgs roscpp std_msgs tf
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

set(SRC_LIST src/estimators/commons.cpp
             src/estimators/sw_estimator.cpp
             src/estimators/integration.cpp
             src/estimators/initial_alignment.cpp
             src/estimators/feature_manager.cpp
             src/estimators/initial_sfm.cpp
             src/estimators/pose_parameterization.cpp
             src/estimators/imu_factor.cpp
             src/estimators/projection_factor.cpp
             src/estimators/marginalization_factor.cpp)

add_executable(${PROJECT_NAME}_node src/estimator_node.cpp ${SRC_LIST})

# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
)


## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
install(TARGETS ${PROJECT_NAME}_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES
  launch/estimator.launch
  config/estimator.yaml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)


