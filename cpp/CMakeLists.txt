cmake_minimum_required(VERSION 3.15)

# YABTE_WHEEL_ONLY builds just the Python extension module, for packaging. It
# needs neither Conan's Arrow (the module links pyarrow's) nor GoogleTest, so
# the whole Arrow/Boost/Thrift dependency chain and the test binary are skipped.
option(YABTE_WHEEL_ONLY "Build only the Python extension module" OFF)

# Default to the in-tree Conan toolchain, but let a caller (setup.py, a preset,
# or -DCMAKE_TOOLCHAIN_FILE=...) supply its own build directory instead.
if(NOT CMAKE_TOOLCHAIN_FILE AND NOT DEFINED ENV{CMAKE_TOOLCHAIN_FILE})
  set(CMAKE_TOOLCHAIN_FILE
      ${CMAKE_SOURCE_DIR}/build/${CMAKE_BUILD_TYPE}/conan_toolchain.cmake)
endif()

project(yabte_cpp)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(glog REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(bshoshany-thread-pool REQUIRED)
find_package(re2 REQUIRED)

if(YABTE_WHEEL_ONLY)
  # Only the module is built, so the interpreter plus Development.Module is
  # enough; requiring Development.Embed would need libpython, which manylinux
  # images deliberately omit.
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
else()
  find_package(Arrow REQUIRED)
  find_package(Python COMPONENTS Interpreter Development REQUIRED)
endif()

# file(GLOB_RECURSE SOURCES src *.cpp)
# file(GLOB_RECURSE SOURCES_TEST src_test *.cpp)
# message(FOO="${SOURCES}")

# add_compile_options(-Wall -Wextra -Wpedantic)
# add_compile_options(-fvisibility=hidden)

if(NOT YABTE_WHEEL_ONLY)

# yabte objects
#
set(YABTE_OBJS yabte_objs)

add_library(
  ${YABTE_OBJS}
  OBJECT
  ${CMAKE_SOURCE_DIR}/src/YABTE/BackTest/Transaction.cpp
  ${CMAKE_SOURCE_DIR}/src/YABTE/BackTest/Asset.cpp
  ${CMAKE_SOURCE_DIR}/src/YABTE/BackTest/Order.cpp
  ${CMAKE_SOURCE_DIR}/src/YABTE/BackTest/PositionalOrder.cpp
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

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  target_compile_definitions(${YABTE_OBJS} PRIVATE YABTE_ENABLE_PROFILING)
endif()

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

endif()

add_subdirectory(pybind)
