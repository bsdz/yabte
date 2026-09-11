
#include "yabte_backtest/test_strategy_01.h"

#include <arrow/compute/api.h>
#include <arrow/io/api.h>
#include <arrow/table.h>
#include <glog/logging.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <vector>

#include "YABTE/BackTest/Strategy.hpp"
#include "YABTE/Utilities/Arrow/TableHelpers.hpp"

using YABTE::BackTest::AssetDataFieldInfo, YABTE::BackTest::SimpleOrder,
    YABTE::BackTest::Strategy;

using YABTE::Utilities::Arrow::ComputeMovingAverage;

using std::optional, std::shared_ptr, std::vector;

shared_ptr<Table> MyExtendTable(const shared_ptr<const Table>& table, int n,
                                int m) {
    auto vals = table->GetColumnByName("('GOOG', 'Close')");
    auto f0 = arrow::field("('GOOG', 'CloseSMAShort')", arrow::float64());
    auto f1 = arrow::field("('GOOG', 'CloseSMALong')", arrow::float64());

    auto st_cma_short = ComputeMovingAverage(vals, n);
    auto st_cma_long = ComputeMovingAverage(vals, m);

    shared_ptr<ChunkedArray> ma_chunked_arr_short = st_cma_short.ValueOrDie();
    shared_ptr<ChunkedArray> ma_chunked_arr_long = st_cma_long.ValueOrDie();

    auto schema = arrow::schema({f0, f1});
    return arrow::Table::Make(schema,
                              {ma_chunked_arr_short, ma_chunked_arr_long});
}

TestSMAXOStrat::TestSMAXOStrat() : Strategy() {}

shared_ptr<Strategy> TestSMAXOStrat::clone() const {
    return std::make_shared<TestSMAXOStrat>(*this);
}

shared_ptr<const Table> TestSMAXOStrat::extend_data(
    const shared_ptr<const Table>& data) {
    auto n = std::get<int>(this->params_.at("n"));
    auto m = std::get<int>(this->params_.at("m"));
    return MyExtendTable(data, n, m);
}

void TestSMAXOStrat::on_open() {
    // can only access this->data_[-1] if in in available-at-open field
    auto asset = (*this->asset_map_)["GOOG"];
    auto fields = asset->_get_fields(AssetDataFieldInfo::AVAILABLE_AT_OPEN);
}

namespace {
// Null-aware scalar read from a (possibly multi-chunk) double column.
optional<double> value_at(const shared_ptr<ChunkedArray>& arr, int64_t i) {
    int64_t offset = 0;
    for (const auto& chunk : arr->chunks()) {
        if (i < offset + chunk->length()) {
            auto d = std::static_pointer_cast<arrow::DoubleArray>(chunk);
            auto local = i - offset;
            if (d->IsNull(local)) return std::nullopt;
            return d->Value(local);
        }
        offset += chunk->length();
    }
    return std::nullopt;
}
}  // namespace

void TestSMAXOStrat::on_close() {
    auto n = std::get<int>(this->params_.at("n"));
    auto m = std::get<int>(this->params_.at("m"));

    // data_ is the full table; the engine tracks position via current_idx_.
    auto idx = this->current_idx_;

    DLOG(INFO) << "on_close";
    if (idx < std::max(n, m) + 1) return;

    auto s_short = this->data_->GetColumnByName("('GOOG', 'CloseSMAShort')");
    auto s_long = this->data_->GetColumnByName("('GOOG', 'CloseSMALong')");

    auto short_prev = value_at(s_short, idx - 1);
    auto short_curr = value_at(s_short, idx);
    auto long_prev = value_at(s_long, idx - 1);
    auto long_curr = value_at(s_long, idx);

    if (!short_prev || !short_curr || !long_prev || !long_curr) return;

    if (*short_prev < *long_prev && *short_curr > *long_curr) {
        this->orders_->push_back(std::make_shared<SimpleOrder>("GOOG", 100));
    } else if (*long_prev < *short_prev && *long_curr > *short_curr) {
        this->orders_->push_back(std::make_shared<SimpleOrder>("GOOG", -100));
    }
}
