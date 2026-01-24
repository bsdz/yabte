#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <pybind11/eval.h>

namespace py = pybind11;

TEST(PythonEmbedTest, SmokeTest) {
    py::scoped_interpreter guard{};
    py::object scope = py::module_::import("__main__").attr("__dict__");
    int result = py::eval("1 + 1", scope).cast<int>();
    ASSERT_EQ(result, 2);
}
