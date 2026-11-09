set(YABTE_CPP_MODULE _yabte_backtest_lib)

# Define core sources directly to recompile them against PyArrow headers
set(YABTE_CORE_SOURCES
    ${CMAKE_SOURCE_DIR}/src/YABTE/BackTest/Transaction.cpp
    ${CMAKE_SOURCE_DIR}/src/YABTE/BackTest/Asset.cpp
    ${CMAKE_SOURCE_DIR}/src/YABTE/BackTest/Order.cpp
    ${CMAKE_SOURCE_DIR}/src/YABTE/BackTest/PositionalOrder.cpp
    ${CMAKE_SOURCE_DIR}/src/YABTE/BackTest/Book.cpp
    ${CMAKE_SOURCE_DIR}/src/YABTE/BackTest/Strategy.cpp
    ${CMAKE_SOURCE_DIR}/src/YABTE/BackTest/StrategyRunner.cpp
    ${CMAKE_SOURCE_DIR}/src/YABTE/Utilities/Arrow/TableHelpers.cpp
)

file(GLOB_RECURSE PYXMOLPP_SOURCES
    yabte_cpp/**.cpp
    yabte_cpp/**.h
)

# Add module with both bindings and core sources
pybind11_add_module(${YABTE_CPP_MODULE} ${PYXMOLPP_SOURCES} ${YABTE_CORE_SOURCES})

# Include directories
target_include_directories(${YABTE_CPP_MODULE} PRIVATE 
    ${CMAKE_CURRENT_LIST_DIR}/yabte_cpp 
    ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(${YABTE_CPP_MODULE} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(${YABTE_CPP_MODULE} PROPERTIES CXX_STANDARD 23)

# Dynamically find PyArrow. The extension links the Arrow libraries shipped in
# the installed pyarrow wheel and compiles against its bundled headers -- see
# the "Two Arrows problem" in AGENTS.md. Arrow's sonames are version-stamped
# (libarrow.so.2300), so the resulting module only loads against the same
# pyarrow minor; keep the pin in pyproject.toml in step.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pyarrow; print(pyarrow.get_include())"
    OUTPUT_VARIABLE PYARROW_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYARROW_PROBE_RESULT
)
if(NOT PYARROW_PROBE_RESULT EQUAL 0 OR PYARROW_INCLUDE_DIR STREQUAL "")
    message(FATAL_ERROR
        "Could not import pyarrow using ${Python_EXECUTABLE}. The extension "
        "needs pyarrow's headers and libraries at build time; install it into "
        "that interpreter, or point CMake at another with "
        "-DPython_EXECUTABLE=/path/to/python.")
endif()

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pyarrow; import os; print(os.path.dirname(pyarrow.__file__))"
    OUTPUT_VARIABLE PYARROW_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Python Executable: ${Python_EXECUTABLE}")
message(STATUS "PyArrow Include: ${PYARROW_INCLUDE_DIR}")
message(STATUS "PyArrow Lib Dir: ${PYARROW_LIB_DIR}")

target_include_directories(${YABTE_CPP_MODULE} PRIVATE ${PYARROW_INCLUDE_DIR})

# Arrow ships as libarrow.so.<major><minor> in the wheel, so glob rather than
# relying on an unversioned symlink (wheels do not ship one).
foreach(_arrow_lib arrow arrow_python parquet)
    file(GLOB _found "${PYARROW_LIB_DIR}/lib${_arrow_lib}.so*")
    if(NOT _found)
        message(FATAL_ERROR
            "lib${_arrow_lib} not found in ${PYARROW_LIB_DIR}")
    endif()
    list(GET _found 0 _lib)
    message(STATUS "Found ${_arrow_lib}: ${_lib}")
    list(APPEND PYARROW_LIBRARIES "${_lib}")
endforeach()

# libparquet is linked explicitly: LoadTable calls parquet::arrow::OpenFile, and
# a shared module would otherwise leave those symbols to be resolved by whatever
# happens to be loaded transitively at import time.
target_link_libraries(${YABTE_CPP_MODULE} PRIVATE
    ${PYARROW_LIBRARIES}
    glog::glog
    bshoshany-thread-pool::bshoshany-thread-pool
    re2::re2
)

# Define YABTE_USE_PYARROW_API to handle API differences in TableHelpers.cpp
target_compile_definitions(${YABTE_CPP_MODULE} PRIVATE YABTE_USE_PYARROW_API)

if(YABTE_WHEEL_ONLY)
    # In an installed wheel the module sits at <site-packages>/yabte/backtest/
    # and pyarrow at <site-packages>/pyarrow/, so resolve the Arrow libraries
    # relative to the module. Absolute build-tree rpaths must not leak into a
    # redistributable wheel.
    set_target_properties(${YABTE_CPP_MODULE} PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "$ORIGIN/../../pyarrow"
    )
    install(TARGETS ${YABTE_CPP_MODULE} DESTINATION yabte/backtest)
else()
    # Development builds load Arrow straight from the build interpreter's wheel.
    target_link_directories(${YABTE_CPP_MODULE} PUBLIC ${PYARROW_LIB_DIR})
endif()
