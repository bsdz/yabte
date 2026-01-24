#include <glog/logging.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <vector>

#include "YABTE/BackTest/Asset.hpp"
#include "YABTE/BackTest/Book.hpp"
#include "YABTE/BackTest/Order.hpp"
#include "YABTE/BackTest/Strategy.hpp"
#include "YABTE/BackTest/StrategyRunner.hpp"
#include "YABTE/Utilities/Arrow/TableHelpers.hpp"
#include "data/test_data.h"
#include "yabte_backtest/test_strategy_01.h"

namespace {
auto death_test_matcher = []() {
    return std::getenv("GTEST_FORCE_DEATH_TEST_FAIL") != nullptr ? "x"s : ".*"s;
};
}  // namespace

using namespace YABTE::BackTest;

using YABTE::Utilities::Arrow::LoadTable;

namespace fs = std::filesystem;

void test_runner_01(bool& success) {
    success = false;

    StrategyVector strategies;
    AssetVector assets;
    BookVector books;

    auto s = TestSMAXOStrat();
    strategies.push_back(std::make_shared<TestSMAXOStrat>(s));

    auto a1 = OHLCAsset("GOOG", "USD");
    assets.push_back(std::make_shared<OHLCAsset>(a1));

    auto b1 = Book("bk1", "USD");
    books.push_back(std::make_shared<Book>(b1));

    auto test_data_dir = fs::weakly_canonical(test_data_path());
    auto np = test_data_dir / "data_sample.parquet";

    auto st_lt = LoadTable(np.string());
    ASSERT_TRUE(st_lt.ok()) << "Error: " << st_lt.status();
    shared_ptr<arrow::Table> table = st_lt.ValueOrDie();

    ASSERT_NO_THROW(({
        auto sr = StrategyRunner(table, assets, strategies, books);
        ParamMap pm;
        pm = {{"n", 10}, {"m", 20}};
        auto srr = sr.run(pm);
        ASSERT_EQ(srr.orders_unprocessed_->size(), 0);
        ASSERT_EQ(srr.orders_processed_.size(), 71);
    }));

    success = true;
}

TEST(RunnerTest, RunSmokeTest) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(
        {
            bool success = false;
            test_runner_01(success);
            if (!success) {
                exit(1);
            }
            exit(0);
        },
        testing::ExitedWithCode(0),
        testing::MatchesRegex(death_test_matcher()));
}