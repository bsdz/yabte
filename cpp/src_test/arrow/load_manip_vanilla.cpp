
#include <arrow/array/data.h>
#include <arrow/compute/api.h>
#include <arrow/io/api.h>
#include <arrow/stl_iterator.h>
#include <arrow/table.h>
#include <gtest/gtest.h>
#include <parquet/arrow/reader.h>

#include <cmath>
#include <filesystem>
#include <memory>
#include <utility>
#include <vector>

#include "YABTE/Utilities/Arrow/TableHelpers.hpp"
#include "data/test_data.h"

using std::shared_ptr, std::vector, std::as_const;

using YABTE::Utilities::Arrow::LoadTable, YABTE::Utilities::Arrow::ReadToTable,
    YABTE::Utilities::Arrow::ExtendTable,
    YABTE::Utilities::Arrow::HorizConcatTables;

namespace fs = std::filesystem;

TEST(ArrowTest, LoadTable) {
    auto test_data_dir = fs::weakly_canonical(test_data_path());
    auto np = test_data_dir / "data_sample.parquet";

    auto st_lt = LoadTable(np.string());
    ASSERT_TRUE(st_lt.ok()) << "Error: " << st_lt.status();
    shared_ptr<arrow::Table> table = st_lt.ValueOrDie();

    auto chunked_array = table->GetColumnByName("('GOOG', 'Close')");

    std::vector<double> values;
    for (auto elem : arrow::stl::Iterate<arrow::DoubleType>(*chunked_array)) {
        if (elem.has_value()) {
            values.push_back(*elem);
        }
    }

    auto &dt = chunked_array->type();
    ASSERT_EQ(dt->id(), arrow::Type::DOUBLE);
    for (auto &chunk : chunked_array->chunks()) {
        // each chunk is an arrow::Array
        auto &data = chunk->data();
        arrow::ArraySpan array_span(*data);
        auto len =
            array_span.buffers[1].size / static_cast<int64_t>(sizeof(double));
        auto raw_values = array_span.GetSpan<double>(1, len);
        // arrow::util::span<const double> raw_values =
        //     data.GetSpan<double>(1);  // 1 is the data buffer
        //  ^ all the scalars of the chunk ara tightly packed here
        //  64 bits for every double even if it's logically NULL
    }
}

TEST(ArrowTest, ExtendTable) {
    auto read_options = arrow::json::ReadOptions::Defaults();
    auto parse_options = arrow::json::ParseOptions::Defaults();
    parse_options.unexpected_field_behavior =
        arrow::json::UnexpectedFieldBehavior::InferType;

    std::string json1 = R"({"a": 1, "b": 2.0, "c": "foo", "d": false}
    {"a": 4, "b": -5.5, "c": null, "d": true}
    )";
    auto st_rtt1 = ReadToTable(json1, read_options, parse_options);
    ASSERT_TRUE(st_rtt1.ok()) << "Error: " << st_rtt1.status();
    auto tab1 = std::dynamic_pointer_cast<arrow::Table>(st_rtt1.ValueOrDie());

    std::string json2 = R"({"e": 3, "f": 4.0}
    {"e": 6, "f": -7.5}
    )";
    auto st_rtt2 = ReadToTable(json2, read_options, parse_options);
    ASSERT_TRUE(st_rtt2.ok()) << "Error: " << st_rtt2.status();
    auto tab2 = std::dynamic_pointer_cast<arrow::Table>(st_rtt2.ValueOrDie());

    auto st_et = ExtendTable(as_const(st_rtt1.ValueOrDie()),
                             as_const(st_rtt2.ValueOrDie()));
    ASSERT_TRUE(st_et.ok()) << "Error: " << st_et.status();
    // order matters!
    shared_ptr<arrow::Table> tab3b = st_et.ValueOrDie();
    auto tab3 = std::dynamic_pointer_cast<arrow::Table>(st_et.ValueOrDie());

    ASSERT_EQ(tab3b->num_columns(), 6);
    ASSERT_EQ(tab3->num_columns(), 6);
}

TEST(ArrowTest, StackTable) {
    auto read_options = arrow::json::ReadOptions::Defaults();
    auto parse_options = arrow::json::ParseOptions::Defaults();
    parse_options.unexpected_field_behavior =
        arrow::json::UnexpectedFieldBehavior::InferType;

    std::string json1 = R"({"a": 1, "b": 2.0, "c": "foo", "d": false}
    {"a": 4, "b": -5.5, "c": null, "d": true}
    )";
    auto st_rtt1 = ReadToTable(json1, read_options, parse_options);
    ASSERT_TRUE(st_rtt1.ok()) << "Error: " << st_rtt1.status();
    auto tab1 = std::dynamic_pointer_cast<arrow::Table>(st_rtt1.ValueOrDie());

    std::string json2 = R"({"e": 3, "f": 4.0}
    {"e": 6, "f": -7.5}
    )";
    auto st_rtt2 = ReadToTable(json2, read_options, parse_options);
    ASSERT_TRUE(st_rtt2.ok()) << "Error: " << st_rtt2.status();
    auto tab2 = std::dynamic_pointer_cast<arrow::Table>(st_rtt2.ValueOrDie());

    std::string json3 = R"({"g": false, "h": 4}
    {"g": true, "h": -5}
    )";
    auto st_rtt3 = ReadToTable(json3, read_options, parse_options);
    ASSERT_TRUE(st_rtt3.ok()) << "Error: " << st_rtt3.status();
    auto tab3 = std::dynamic_pointer_cast<arrow::Table>(st_rtt3.ValueOrDie());

    vector<shared_ptr<const arrow::Table>> tables = {tab1, tab2, tab3};
    vector<string> field_prefixes = {"tab1_", "tab2_", "tab3_"};

    auto st_et = HorizConcatTables(tables, field_prefixes);
    ASSERT_TRUE(st_et.ok()) << "Error: " << st_et.status();
    // order matters!
    shared_ptr<arrow::Table> tabxb = st_et.ValueOrDie();
    auto tabx = std::dynamic_pointer_cast<arrow::Table>(st_et.ValueOrDie());

    ASSERT_EQ(tabxb->num_columns(), 8);
    ASSERT_EQ(tabx->num_columns(), 8);
}
