
find_package(GTest REQUIRED)

add_subdirectory(smoke_test_apps)

enable_testing()

# gtest executable
#
set(GTEST_YABTE_EXE gtest_yabte)

add_executable(
    ${GTEST_YABTE_EXE}
    ${CMAKE_SOURCE_DIR}/src_test/data/test_data.cpp
    ${CMAKE_SOURCE_DIR}/src_test/pybind/embed_vanilla.cpp
    ${CMAKE_SOURCE_DIR}/src_test/yabte_backtest/objects.cpp
    ${CMAKE_SOURCE_DIR}/src_test/yabte_backtest/test_strategy_01.cpp

    ${CMAKE_SOURCE_DIR}/src_test/yabte_backtest/strategy_runner_run.cpp

    ${CMAKE_SOURCE_DIR}/src_test/yabte_backtest/strategy_runner_optimize.cpp
    ${CMAKE_SOURCE_DIR}/src_test/arrow/load_manip_vanilla.cpp
)
target_link_libraries(
    ${GTEST_YABTE_EXE}
    ${YABTE_LIB_STATIC}
    GTest::gtest_main
)
target_include_directories(
    ${GTEST_YABTE_EXE}
    PRIVATE ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${CMAKE_SOURCE_DIR}/src_test
)

# test discovery
#
include(CTest)
include(GoogleTest)
gtest_discover_tests(${GTEST_YABTE_EXE})
