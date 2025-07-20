set(YABTE_CPP_MODULE yabte_cpp_backtest)

file(GLOB_RECURSE PYXMOLPP_SOURCES
    yabte_cpp/**.cpp
    yabte_cpp/**.h
)
pybind11_add_module(${YABTE_CPP_MODULE} ${PYXMOLPP_SOURCES})
target_include_directories(${YABTE_CPP_MODULE} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/yabte_cpp PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(${YABTE_CPP_MODULE} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
target_link_libraries(${YABTE_CPP_MODULE} PRIVATE ${YABTE_LIB_STATIC})

set(VENV_DIR /home/blair/projects/yabte_cpp/.venv)
target_include_directories(${YABTE_CPP_MODULE} PUBLIC ${VENV_DIR}/lib/python3.12/site-packages/pyarrow/include)
target_link_directories(${YABTE_CPP_MODULE} PUBLIC ${VENV_DIR}/lib/python3.12/site-packages/pyarrow)
target_link_libraries(${YABTE_CPP_MODULE} PRIVATE arrow_python)

# target_compile_definitions(
# ${YABTE_CPP_MODULE}
# PUBLIC PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF
# )