cmake_minimum_required(VERSION 3.16)
project(libneupan_uav VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(NEUPAN_UAV_BUILD_TESTS "Build libneupan_uav unit tests" ON)
option(NEUPAN_UAV_ENABLE_RKNN "Build RKNN runtime support when available" ON)

find_package(Eigen3 3.3 REQUIRED)
find_package(osqp CONFIG QUIET)
if(NOT osqp_FOUND)
  message(FATAL_ERROR
    "OSQP is required by libneupan_uav. "
    "Set osqp_DIR or CMAKE_PREFIX_PATH to the OSQP installation.")
endif()
if(TARGET osqp::osqp)
  set(NEUPAN_UAV_OSQP_TARGET osqp::osqp)
elseif(TARGET osqp::osqpstatic)
  set(NEUPAN_UAV_OSQP_TARGET osqp::osqpstatic)
elseif(TARGET osqp)
  set(NEUPAN_UAV_OSQP_TARGET osqp)
else()
  message(FATAL_ERROR
    "OSQP was found, but it did not export a supported target. "
    "Expected osqp::osqp, osqp::osqpstatic, or osqp.")
endif()

if(NEUPAN_UAV_ENABLE_RKNN)
  find_path(RKNN_INCLUDE_DIR
    NAMES rknn_api.h
    PATHS
      /usr/local/include
      /usr/include
      /home/orangepi/third_party/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/include
  )
  find_library(RKNNRT_LIBRARY
    NAMES rknnrt librknnrt
    PATHS /lib /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu
  )
  if(RKNN_INCLUDE_DIR AND RKNNRT_LIBRARY)
    set(NEUPAN_UAV_RKNN_FOUND ON)
    message(STATUS "Found RKNN runtime: ${RKNNRT_LIBRARY}")
    message(STATUS "Found RKNN include: ${RKNN_INCLUDE_DIR}")
  else()
    message(WARNING "RKNN header/runtime was not found; RKNN runtime runner will be disabled")
  endif()
endif()

add_library(neupan_uav
  src/config.cpp
  src/robot_model.cpp
  src/farfield_guide.cpp
  src/obstacle_preselector.cpp
  src/dune.cpp
  src/rknn_runner.cpp
  src/nrmp.cpp
  src/pan.cpp
  src/planner.cpp
)
add_library(neupan_uav::neupan_uav ALIAS neupan_uav)

target_include_directories(neupan_uav PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(neupan_uav
  PUBLIC Eigen3::Eigen
  PRIVATE ${NEUPAN_UAV_OSQP_TARGET}
)
target_compile_options(neupan_uav PRIVATE -Wall -Wextra -Wpedantic)

if(NEUPAN_UAV_RKNN_FOUND)
  target_compile_definitions(neupan_uav PUBLIC NEUPAN_UAV_WITH_RKNN)
  target_include_directories(neupan_uav PUBLIC ${RKNN_INCLUDE_DIR})
  target_link_libraries(neupan_uav PUBLIC ${RKNNRT_LIBRARY})
endif()

if(NEUPAN_UAV_BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)

  foreach(t
      test_config
      test_warm_start
      test_farfield_guide
      test_obstacle_preselector
      test_dune_postprocess
      test_rknn_runner
      test_nrmp
      test_replay)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE
      neupan_uav GTest::gtest GTest::gtest_main)
    add_test(NAME ${t} COMMAND ${t})
  endforeach()
  target_compile_definitions(test_dune_postprocess PRIVATE
    NEUPAN_UAV_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")

  if(NEUPAN_UAV_RKNN_FOUND)
    add_executable(dump_rknn_raw_mu tests/tools/dump_rknn_raw_mu.cpp)
    target_link_libraries(dump_rknn_raw_mu PRIVATE neupan_uav)
  endif()

  add_executable(run_phase6_5_cpp_replay
    tests/tools/run_phase6_5_cpp_replay.cpp)
  target_link_libraries(run_phase6_5_cpp_replay PRIVATE neupan_uav)
endif()

install(TARGETS neupan_uav EXPORT neupan_uavTargets)
install(DIRECTORY include/ DESTINATION include)
install(FILES package.xml DESTINATION share/libneupan_uav)

install(EXPORT neupan_uavTargets
  FILE libneupan_uavTargets.cmake
  NAMESPACE neupan_uav::
  DESTINATION lib/cmake/libneupan_uav)

include(CMakePackageConfigHelpers)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/libneupan_uavConfig.cmake" "\
include(CMakeFindDependencyMacro)
find_dependency(Eigen3)
find_dependency(osqp CONFIG)
include(\"\${CMAKE_CURRENT_LIST_DIR}/libneupan_uavTargets.cmake\")
")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libneupan_uavConfig.cmake"
  DESTINATION lib/cmake/libneupan_uav)
