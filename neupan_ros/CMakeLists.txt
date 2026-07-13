cmake_minimum_required(VERSION 3.16)
project(neupan_ros VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(mavros_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(yaml_cpp_vendor REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(libneupan_uav REQUIRED)

add_library(neupan_ros_core
  src/adapters.cpp
  src/config_loader.cpp
  src/px4_control_logic.cpp
)
target_include_directories(neupan_ros_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(neupan_ros_core
  Eigen3::Eigen
  yaml-cpp
  neupan_uav::neupan_uav
)
ament_target_dependencies(neupan_ros_core
  geometry_msgs
  mavros_msgs
  nav_msgs
  rclcpp
  sensor_msgs
  std_msgs
)
target_compile_options(neupan_ros_core PRIVATE -Wall -Wextra -Wpedantic)

add_library(neupan_ros_nodes
  src/uav_node.cpp
  src/px4_control.cpp
)
target_compile_definitions(neupan_ros_nodes PRIVATE NEUPAN_ROS_DISABLE_NODE_MAIN)
target_link_libraries(neupan_ros_nodes neupan_ros_core)
ament_target_dependencies(neupan_ros_nodes
  geometry_msgs
  mavros_msgs
  nav_msgs
  rclcpp
  sensor_msgs
  std_msgs
)
target_compile_options(neupan_ros_nodes PRIVATE -Wall -Wextra -Wpedantic)

add_executable(neupan_uav_node
  src/uav_node.cpp
)
target_link_libraries(neupan_uav_node neupan_ros_core)
ament_target_dependencies(neupan_uav_node rclcpp)
target_compile_options(neupan_uav_node PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(neupan_uav_node PROPERTIES
  INSTALL_RPATH "/usr/local/lib;/usr/lib;/lib"
)

add_executable(px4_control
  src/px4_control.cpp
)
target_link_libraries(px4_control neupan_ros_core)
ament_target_dependencies(px4_control rclcpp mavros_msgs)
target_compile_options(px4_control PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(px4_control PROPERTIES
  INSTALL_RPATH "/usr/local/lib;/usr/lib;/lib"
)

install(TARGETS neupan_ros_core neupan_uav_node px4_control
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config models DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_adapters test/test_adapters.cpp)
  target_link_libraries(test_adapters neupan_ros_core)

  ament_add_gtest(test_config_loader test/test_config_loader.cpp)
  target_link_libraries(test_config_loader neupan_ros_core)
  target_compile_definitions(test_config_loader PRIVATE
    NEUPAN_ROS_TEST_CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/config")

  ament_add_gtest(test_px4_control_logic test/test_px4_control_logic.cpp)
  target_link_libraries(test_px4_control_logic neupan_ros_core)

  ament_add_gtest(test_px4_control_topics test/test_px4_control_topics.cpp)
  target_link_libraries(test_px4_control_topics neupan_ros_nodes)

  ament_add_gtest(test_uav_node_smoke test/test_uav_node_smoke.cpp)
  target_link_libraries(test_uav_node_smoke neupan_ros_nodes)
  target_compile_definitions(test_uav_node_smoke PRIVATE
    NEUPAN_ROS_TEST_CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/config"
    NEUPAN_ROS_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test")
endif()

ament_package()
