cmake_minimum_required(VERSION 3.15)
set(CMAKE_TOOLCHAIN_FILE ${CMAKE_SOURCE_DIR}/build/${CMAKE_BUILD_TYPE}/conan_toolchain.cmake)
project(yabte_cpp)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Arrow REQUIRED)
find_package(glog REQUIRED)
find_package(Python 3.12 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(bshoshany-thread-pool REQUIRED)

# file(GLOB_RECURSE SOURCES src *.cpp)
# file(GLOB_RECURSE SOURCES_TEST src_test *.cpp)
# message(FOO="${SOURCES}")

# add_compile_options(-Wall -Wextra -Wpedantic)
# add_compile_options(-fvisibility=hidden)

# yabte objects
#
set(YABTE_OBJS yabte_objs)

add_library(
  ${YABTE_OBJS}
  OBJECT
  ${CMAKE_SOURCE_DIR}/src/YABTE/BackTest/Transaction.cpp
  ${CMAKE_SOURCE_DIR}/src/YABTE/BackTest/Asset.cpp
  ${CMAKE_SOURCE_DIR}/src/YABTE/BackTest/Order.cpp
  ${CMAKE_SOURCE_DIR}/src/YABTE/BackTest/Book.cpp
  ${CMAKE_SOURCE_DIR}/src/YABTE/BackTest/Strategy.cpp
  ${CMAKE_SOURCE_DIR}/src/YABTE/BackTest/StrategyRunner.cpp
  ${CMAKE_SOURCE_DIR}/src/YABTE/Utilities/Arrow/TableHelpers.cpp
)

# shared libraries will need PIC. for later performance, we can use
# separate shared and static objects but will double compile time
set_property(TARGET ${YABTE_OBJS} PROPERTY POSITION_INDEPENDENT_CODE 1)

target_link_libraries(
  ${YABTE_OBJS}
  arrow::arrow
  glog::glog
  pybind11::embed
  bshoshany-thread-pool::bshoshany-thread-pool
)

target_include_directories(
  ${YABTE_OBJS}
  PUBLIC ${CMAKE_SOURCE_DIR}/include
)

# target_compile_definitions(
# ${YABTE_OBJS}
# PUBLIC PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF
# )

# yabte library static & shared
#
set(YABTE_LIB_STATIC yabte_lib_static)

add_library(
  ${YABTE_LIB_STATIC}
  STATIC
  $<TARGET_OBJECTS:${YABTE_OBJS}>
)
target_link_libraries(
  ${YABTE_LIB_STATIC}
  arrow::arrow
  glog::glog
  pybind11::embed
  bshoshany-thread-pool::bshoshany-thread-pool
)

set(YABTE_LIB_SHARED yabte_lib_shared)

add_library(
  ${YABTE_LIB_SHARED}
  SHARED
  $<TARGET_OBJECTS:${YABTE_OBJS}>
)

add_subdirectory(src_test)

add_subdirectory(pybind)
