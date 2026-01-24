#pragma once

#include <glog/logging.h>

#include <exception>
#include <filesystem>
#include <memory>
#include <ranges>
#include <vector>

#include "YABTE/BackTest/Asset.hpp"
#include "YABTE/BackTest/Book.hpp"
#include "YABTE/BackTest/Order.hpp"
#include "YABTE/BackTest/Strategy.hpp"
#include "YABTE/BackTest/StrategyRunner.hpp"
#include "YABTE/Utilities/Arrow/TableHelpers.hpp"
#include "data/test_data.h"
#include "yabte_backtest/test_strategy_01.h"

using namespace YABTE::BackTest;

using YABTE::Utilities::Arrow::LoadTable;

using std::shared_ptr, std::vector, std::exception;

namespace fs = std::filesystem;

int test_optimize_01() {
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
    CHECK(st_lt.ok()) << "Error: " << st_lt.status();
    shared_ptr<arrow::Table> table = st_lt.ValueOrDie();

    std::vector<int> seq1{20, 30, 40, 50};
    std::vector<int> seq2{5, 10, 15, 20};

    vector<ParamMap> param_vector;
    for (const auto& [n, m] : std::views::zip(seq1, seq2)) {
        if (n <= m) continue;
        param_vector.push_back(ParamMap({{"n"s, n}, {"m"s, m}}));
    }

    try {
        auto sr = StrategyRunner(table, assets, strategies, books);
        auto srrs = sr.run_batch(param_vector, 2);
    } catch (exception& e) {
        LOG(ERROR) << "strategy optimizer failed: " << e.what();
        return -1;
    } catch (...) {
        LOG(ERROR) << "strategy optimizer failed unknown reason";
        return -1;
    }

    return 0;
}
