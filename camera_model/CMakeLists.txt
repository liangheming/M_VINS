cmake_minimum_required(VERSION 3.0.2)
project(camera_model)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC")


find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

catkin_package(
 INCLUDE_DIRS src
 LIBRARIES camera_model
 CATKIN_DEPENDS roscpp std_msgs
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

# Declare a C++ library
add_library(${PROJECT_NAME}
  src/camera_model/pinhole_camera.cpp
)

target_link_libraries(${PROJECT_NAME}  ${OpenCV_LIBS})


## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

## Mark cpp header files for installation
install(DIRECTORY src/camera_model/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  # PATTERN ".svn" EXCLUDE
)