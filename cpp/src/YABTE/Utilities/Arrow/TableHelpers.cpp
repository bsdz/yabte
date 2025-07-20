#include "YABTE/Utilities/Arrow/TableHelpers.hpp"

#include <arrow/builder.h>
#include <arrow/compute/api.h>
#include <arrow/json/reader.h>
#include <glog/logging.h>
#include <parquet/arrow/reader.h>

#include <cmath>
#include <ranges>

using std::dynamic_pointer_cast, std::exception, std::shared_ptr, std::vector,
    std::views::iota, std::string_literals::operator""s, std::make_shared,
    std::string;

namespace YABTE::Utilities::Arrow {

Result<shared_ptr<Table>> LoadTable(string path) {
    arrow::MemoryPool* pool = arrow::default_memory_pool();
    shared_ptr<arrow::io::RandomAccessFile> input;
    ARROW_ASSIGN_OR_RAISE(input, arrow::io::ReadableFile::Open(path));

    // Open Parquet file reader
    std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
    ARROW_RETURN_NOT_OK(parquet::arrow::OpenFile(input, pool, &arrow_reader));

    // Read entire file as a single Arrow table
    shared_ptr<Table> table;
    ARROW_RETURN_NOT_OK(arrow_reader->ReadTable(&table));
    return table;
}

Result<shared_ptr<ChunkedArray>> ComputeMovingAverage(
    shared_ptr<ChunkedArray> vals, int n) {
    // calculate moving average vector
    vector<double> ma(n, std::nan(""));

    // rows 0..9, 1..10, 2..11, ..., size-n-1..size-1
    for (int x : iota(0, vals->length() - n)) {
        auto slice = vals->Slice(x, n);
        arrow::Datum sum;
        ARROW_ASSIGN_OR_RAISE(sum, arrow::compute::Mean({slice}));
        ma.push_back(sum.scalar_as<arrow::DoubleScalar>().value);
    }

    // convert vector to arrow array
    shared_ptr<arrow::Array> ma_arr;
    arrow::DoubleBuilder dbl_builder = arrow::DoubleBuilder();

    ARROW_RETURN_NOT_OK(dbl_builder.AppendValues(ma.begin(), ma.end()));
    ARROW_ASSIGN_OR_RAISE(ma_arr, dbl_builder.Finish());

    return make_shared<ChunkedArray>(ma_arr);
}

// NOTE: This function assumes the first column of both tables are identical.
//       More specifically, this is a naive join in the simplest scenario.
Result<shared_ptr<Table>> ExtendTable(
    const shared_ptr<const Table>& base_table,
    const shared_ptr<const Table>& ext_table) {
    // merge schema metadata from both tables
    auto ext_schema = ext_table->schema();
    auto base_schema = base_table->schema();

    shared_ptr<const arrow::KeyValueMetadata> base_schema_meta;
    shared_ptr<const arrow::KeyValueMetadata> combined_meta;
    if (base_schema->HasMetadata() && ext_schema->HasMetadata()) {
        base_schema_meta = base_schema->metadata()->Copy();
        combined_meta = base_schema_meta->Merge(*ext_schema->metadata());
    }

    // Concatenate schema fields
    arrow::FieldVector combined_fields{base_schema->fields()};
    combined_fields.insert(combined_fields.end()
                           // do this if we assume the first col is identical
                           // ,++ext_schema->fields().cbegin()
                           ,
                           ext_schema->fields().cbegin(),
                           ext_schema->fields().cend());

    // Concatenate table columns
    arrow::ChunkedArrayVector combined_cols{base_table->columns()};
    combined_cols.insert(combined_cols.end()
                         // do this if we assume the first col is identical
                         // ,++ext_table->columns().cbegin()
                         ,
                         ext_table->columns().cbegin(),
                         ext_table->columns().cend());

    // Construct new table from concatenated schema fields and table columns
    return Table::Make(arrow::schema(combined_fields, combined_meta),
                       combined_cols);
}

Result<shared_ptr<Table>> HorizConcatTables(
    vector<shared_ptr<const Table>>& tables, vector<string>& field_prefixes) {
    if (tables.size() != field_prefixes.size()) {
        return Status::Invalid(
            "Tables and field prefixes must have the same size.");
    }

    auto comb = [](auto pf, auto name) {
        return "('" + pf + "', '" + name + "')";
    };

    arrow::ChunkedArrayVector combined_cols;
    arrow::SchemaBuilder schema_builder;
    for (auto const& [pf, table] : std::views::zip(field_prefixes, tables)) {
        if (table->num_rows() != tables[0]->num_rows()) {
            return Status::Invalid("Tables must have the same number of rows.");
        }

        for (auto const& [field, col] :
             std::views::zip(table->fields(), table->columns())) {
            combined_cols.push_back(col);
            ARROW_RETURN_NOT_OK(schema_builder.AddField(
                field->WithName(comb(pf, field->name()))));
        }
    }

    std::shared_ptr<arrow::Schema> schema;
    ARROW_ASSIGN_OR_RAISE(schema, schema_builder.Finish());

    return Table::Make(schema, combined_cols);
}

// see arrow's cpp/src/arrow/json/reader_test.cc
inline static Status MakeStream(std::string_view src_str,
                                shared_ptr<arrow::io::InputStream>* out) {
    auto src = make_shared<arrow::Buffer>(src_str);
    *out = make_shared<arrow::io::BufferReader>(src);
    return Status::OK();
}

Result<shared_ptr<Table>> ReadToTable(
    string json, const arrow::json::ReadOptions& read_options,
    const arrow::json::ParseOptions& parse_options) {
    shared_ptr<arrow::io::InputStream> input;
    RETURN_NOT_OK(MakeStream(json, &input));
    ARROW_ASSIGN_OR_RAISE(auto reader, arrow::json::TableReader::Make(
                                           arrow::default_memory_pool(), input,
                                           read_options, parse_options));
    return reader->Read();
}

}  // namespace YABTE::Utilities::Arrow
