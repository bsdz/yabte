#include "yabte_backtest/strategy_runner_optimize.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <arrow/compute/initialize.h>

namespace py = pybind11;

using std::string_literals::operator""s;

namespace {
auto death_test_matcher = []() {
    return std::getenv("GTEST_FORCE_DEATH_TEST_FAIL") != nullptr ? "x"s : ".*"s;
};
}  // namespace

TEST(RunnerTest, OptimizeSmokeTest) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(
        {
            py::scoped_interpreter guard{};
            py::module_::import("pyarrow");

            // Explicitly initialize Arrow Compute to register all functions
            arrow::compute::Initialize();

            auto res = test_optimize_01();
            if (res) {
                exit(1);
            }
            exit(0);
        },
        testing::ExitedWithCode(0),
        testing::MatchesRegex(death_test_matcher()));
}