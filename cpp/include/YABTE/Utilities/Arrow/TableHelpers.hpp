#pragma once

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/json/options.h>
#include <arrow/stl.h>
#include <arrow/table.h>
#include <arrow/type_traits.h>

#include <chrono>
#include <memory>
#include <vector>

using std::shared_ptr, std::string, std::vector, std::chrono::system_clock,
    std::chrono::time_point, std::chrono::nanoseconds;

using arrow::Status, arrow::Result, arrow::Table, arrow::ChunkedArray,
    arrow::TypeTraits, arrow::TimestampType;

namespace arrow {

template <>
struct CTypeTraits<time_point<system_clock>> {
    using ArrowType = ::arrow::TimestampType;

    static std::shared_ptr<::arrow::DataType> type_singleton() {
        return ::arrow::timestamp(::arrow::TimeUnit::NANO);
    }
};

}  // namespace arrow

namespace arrow::stl {

template <>
struct ConversionTraits<time_point<system_clock>>
    : public CTypeTraits<time_point<system_clock>> {
    constexpr static bool nullable = false;

    static Status AppendRow(
        typename TypeTraits<TimestampType>::BuilderType &builder,
        time_point<system_clock> cell) {
        return builder.Append(
            duration_cast<nanoseconds>(cell.time_since_epoch()).count());
    }

    static time_point<system_clock> GetEntry(const TimestampArray &array,
                                             size_t j) {
        return time_point<system_clock>(nanoseconds(array.Value(j)));
    }
};
}  // namespace arrow::stl

namespace YABTE::Utilities::Arrow {

Result<shared_ptr<Table>> LoadTable(string path);

Result<shared_ptr<ChunkedArray>> ComputeMovingAverage(
    shared_ptr<ChunkedArray> vals, int n);

Result<shared_ptr<Table>> ExtendTable(const shared_ptr<const Table> &base_table,
                                      const shared_ptr<const Table> &ext_table);

Result<shared_ptr<Table>> HorizConcatTables(
    vector<shared_ptr<const Table>> &tables, vector<string> &field_prefixes);

Result<shared_ptr<Table>> ReadToTable(
    string json, const arrow::json::ReadOptions &read_options,
    const arrow::json::ParseOptions &parse_options);

}  // namespace YABTE::Utilities::Arrow