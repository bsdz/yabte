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

# Dynamically find PyArrow
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pyarrow; print(pyarrow.get_include())"
    OUTPUT_VARIABLE PYARROW_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Find PyArrow library directory (where libarrow_python.so lives)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pyarrow; import os; print(os.path.dirname(pyarrow.__file__))"
    OUTPUT_VARIABLE PYARROW_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Python Executable: ${Python_EXECUTABLE}")
message(STATUS "PyArrow Include: ${PYARROW_INCLUDE_DIR}")
message(STATUS "PyArrow Lib Dir: ${PYARROW_LIB_DIR}")

# Add PyArrow headers. PUBLIC to ensure they are visible.
target_include_directories(${YABTE_CPP_MODULE} PUBLIC ${PYARROW_INCLUDE_DIR})

# Find libraries in the PyArrow directory
# Note: In newer PyArrow wheels, libarrow.so might be libarrow.so.2300 or similar
# We can find it by searching for libarrow.so*
file(GLOB ARROW_LIB_FILES "${PYARROW_LIB_DIR}/libarrow.so*")
list(GET ARROW_LIB_FILES 0 ARROW_LIB)

find_library(ARROW_PYTHON_LIB arrow_python HINTS ${PYARROW_LIB_DIR} NO_DEFAULT_PATH)

# Find re2 from Conan to link it statically if needed
find_package(re2 REQUIRED)
find_package(Snappy REQUIRED)

message(STATUS "Found Arrow Lib: ${ARROW_LIB}")
message(STATUS "Found Arrow Python Lib: ${ARROW_PYTHON_LIB}")

# Link against the wheel's libraries + utilities
target_link_libraries(${YABTE_CPP_MODULE} PRIVATE 
    ${ARROW_PYTHON_LIB}
    ${ARROW_LIB}
    glog::glog
    bshoshany-thread-pool::bshoshany-thread-pool
    re2::re2
    Snappy::snappy
)

# Define YABTE_USE_PYARROW_API to handle API differences in TableHelpers.cpp
target_compile_definitions(${YABTE_CPP_MODULE} PRIVATE YABTE_USE_PYARROW_API)

# Explicitly add rpath so the .so can find libarrow.so at runtime
target_link_directories(${YABTE_CPP_MODULE} PUBLIC ${PYARROW_LIB_DIR})

