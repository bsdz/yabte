
#include "yabte_backtest/test_strategy_01.h"

#include <arrow/compute/api.h>
#include <arrow/io/api.h>
#include <arrow/table.h>
#include <glog/logging.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "YABTE/BackTest/Strategy.hpp"
#include "YABTE/Utilities/Arrow/TableHelpers.hpp"

using YABTE::BackTest::AssetDataFieldInfo, YABTE::BackTest::SimpleOrder,
    YABTE::BackTest::Strategy;

using YABTE::Utilities::Arrow::ComputeMovingAverage;

using std::shared_ptr, std::vector;

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

void TestSMAXOStrat::on_close() {
    auto asset = (*this->asset_map_)["GOOG"];

    auto n = std::get<int>(this->params_.at("n"));
    auto m = std::get<int>(this->params_.at("m"));

    auto nrows = this->data_->num_rows();

    DLOG(INFO) << "on_close";
    if (nrows >= std::max(n, m) + 2) {
        auto s_short =
            this->data_->GetColumnByName("('GOOG', 'CloseSMAShort')");
        auto s_long = this->data_->GetColumnByName("('GOOG', 'CloseSMALong')");

        std::vector<optional<double>> v_short{
            arrow::stl::End<arrow::DoubleType>(*s_short) - 2,
            arrow::stl::End<arrow::DoubleType>(*s_short)};
        std::vector<optional<double>> v_long{
            arrow::stl::End<arrow::DoubleType>(*s_long) - 2,
            arrow::stl::End<arrow::DoubleType>(*s_long)};

        if (v_short[0].value() < v_long[0].value() &&
            v_short[1].value() > v_long[1].value()) {
            this->orders_->push_back(
                std::make_shared<SimpleOrder>("GOOG", 100));
        } else if (v_long[0].value() < v_short[0].value() &&
                   v_long[1].value() > v_short[1].value()) {
            this->orders_->push_back(
                std::make_shared<SimpleOrder>("GOOG", -100));
        }
    }
}
