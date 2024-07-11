cmake_minimum_required(VERSION 3.8)
project(odom_recorder)

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
find_package(nav_msgs REQUIRED)

add_executable(${PROJECT_NAME}_node src/${PROJECT_NAME}_node.cpp)
ament_target_dependencies(${PROJECT_NAME}_node rclcpp std_msgs nav_msgs)

install(TARGETS ${PROJECT_NAME}_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})


ament_package()
