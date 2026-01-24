# py subinterpreter scratch executable
#
set(SMOKE_PYSUBINT_EXE smokeapp_pysubint)

add_executable(
    ${SMOKE_PYSUBINT_EXE}
    ${CMAKE_SOURCE_DIR}/src_test/smoke_test_apps/py_subint.cpp
)
target_link_libraries(
    ${SMOKE_PYSUBINT_EXE}
    glog::glog
    pybind11::embed
)
target_include_directories(
    ${SMOKE_PYSUBINT_EXE}
    PRIVATE ${CMAKE_SOURCE_DIR}/include
)

# yabte MT scratch executable
#
set(SMOKE_YABTE_MT_EXE smokeapp_yabte_mt)

add_executable(
    ${SMOKE_YABTE_MT_EXE}
    ${CMAKE_SOURCE_DIR}/src_test/data/test_data.cpp
    ${CMAKE_SOURCE_DIR}/src_test/yabte_backtest/test_strategy_01.cpp
    ${CMAKE_SOURCE_DIR}/src_test/smoke_test_apps/yabte_mt.cpp
)
target_link_libraries(
    ${SMOKE_YABTE_MT_EXE}
    ${YABTE_LIB_STATIC}
)
target_include_directories(
    ${SMOKE_YABTE_MT_EXE}
    PRIVATE ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${CMAKE_SOURCE_DIR}/src_test
)