cmake_minimum_required(VERSION 3.0.2)
project(pose_graph)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  cv_bridge
  camera_model
  sensor_msgs
  nav_msgs
  geometry_msgs
)

find_package(OpenCV)
find_package(Ceres REQUIRED)


catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
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
             src/pose_graph/key_frame.cpp
             src/pose_graph/pose_graph_4dof.cpp
             )
add_executable(${PROJECT_NAME}_node src/pose_graph_node.cpp ${SRC_LIST})

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
  launch/pose_graph.launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

