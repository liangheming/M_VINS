cmake_minimum_required(VERSION 3.8)
project(pose_graph)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-O3)
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)


include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${Ceres_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)


set(SRC_LIST src/ThirdParty/DBoW/BowVector.cpp
             src/ThirdParty/DBoW/FBrief.cpp
             src/ThirdParty/DBoW/FeatureVector.cpp
             src/ThirdParty/DBoW/QueryResults.cpp
             src/ThirdParty/DBoW/ScoringObject.cpp
             src/ThirdParty/DUtils/Random.cpp
             src/ThirdParty/DUtils/Timestamp.cpp
             src/ThirdParty/DVision/BRIEF.cpp
             src/ThirdParty/VocabularyBinary.cpp
             src/pose_graph/commons.cpp
             src/pose_graph/pinhole_camera.cpp
             src/pose_graph/key_frame.cpp
             src/pose_graph/pose_graph_4dof.cpp
             )
add_executable(${PROJECT_NAME}_node src/${PROJECT_NAME}_node.cpp ${SRC_LIST})
ament_target_dependencies(${PROJECT_NAME}_node rclcpp std_msgs nav_msgs sensor_msgs geometry_msgs)
target_link_libraries(${PROJECT_NAME}_node
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  yaml-cpp
)

ament_package()
