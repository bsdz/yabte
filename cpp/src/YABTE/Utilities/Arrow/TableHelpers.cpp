#include "YABTE/Utilities/Arrow/TableHelpers.hpp"

#include <arrow/builder.h>
#include <arrow/json/reader.h>
#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>
#include <cmath>
#include <vector>
#include <ranges>

using namespace std;

namespace YABTE::Utilities::Arrow {

Result<shared_ptr<Table>> LoadTable(string path) {
    shared_ptr<::arrow::io::ReadableFile> infile;
    ARROW_ASSIGN_OR_RAISE(infile, ::arrow::io::ReadableFile::Open(path));

    std::unique_ptr<::parquet::arrow::FileReader> reader;
    ARROW_ASSIGN_OR_RAISE(reader, ::parquet::arrow::OpenFile(infile, ::arrow::default_memory_pool()));

    shared_ptr<Table> table;
    ARROW_RETURN_NOT_OK(reader->ReadTable(&table));
    return table;
}

Result<shared_ptr<ChunkedArray>> ComputeMovingAverage(

    shared_ptr<ChunkedArray> vals, int n) {
    
    if (vals->length() == 0) {
        return vals;
    }

    // 1. Flatten ChunkedArray to a single vector for fast O(1) access
    // This avoids overhead of handling chunks inside the tight loop
    std::vector<double> data;
    data.reserve(vals->length());

    for (const auto& chunk : vals->chunks()) {
        auto d_array = std::dynamic_pointer_cast<::arrow::DoubleArray>(chunk);
        if (!d_array) {
            return Status::Invalid("Input array must be of type Double for Moving Average");
        }
        
        // Use raw_values for maximum speed
        const double* values = d_array->raw_values();
        // Handle nulls if necessary (here we assume strict doubles or treat null as NaN)
        if (d_array->null_count() > 0) {
            for(int64_t i=0; i<d_array->length(); ++i) {
                if (d_array->IsNull(i)) data.push_back(std::nan(""));
                else data.push_back(values[i]);
            }
        } else {
             data.insert(data.end(), values, values + d_array->length());
        }
    }

    // 2. O(N) Sliding Window Algorithm
    std::vector<double> ma;
    ma.reserve(data.size());

    if (data.size() < n) {
         for(size_t i=0; i<data.size(); ++i) ma.push_back(std::nan(""));
    } else {
        // Pad the initial n-1 elements with NaN
        for(int i=0; i<n-1; ++i) ma.push_back(std::nan(""));

        // Compute first window sum (indices 0 to n-1)
        double sum = 0.0;
        for(int i=0; i<n; ++i) {
            sum += data[i]; 
        }
        ma.push_back(sum / n);

        // Slide the window
        for (size_t i = n; i < data.size(); ++i) {
            sum -= data[i - n]; // Remove outgoing
            sum += data[i];     // Add incoming
            ma.push_back(sum / n);
        }
    }

    // 3. Convert back to Arrow
    ::arrow::DoubleBuilder dbl_builder;
    ARROW_RETURN_NOT_OK(dbl_builder.Reserve(ma.size()));

    // Create a validity bitmap or append manually checking for NaN
    // Since we want O(N), we can iterate and append. 
    // DoubleBuilder::AppendValues with validity bitmap is faster but requires creating the bitmap first.
    
    // Let's create validity bitmap manually
    std::vector<uint8_t> valid_bytes(ma.size());
    for(size_t i=0; i<ma.size(); ++i) {
        valid_bytes[i] = std::isnan(ma[i]) ? 0 : 1;
    }
    
    ARROW_RETURN_NOT_OK(dbl_builder.AppendValues(ma.data(), ma.size(), valid_bytes.data()));
    
    std::shared_ptr<::arrow::Array> ma_arr;
    ARROW_ASSIGN_OR_RAISE(ma_arr, dbl_builder.Finish());

    return std::make_shared<::arrow::ChunkedArray>(ma_arr);
}

// NOTE: This function assumes the first column of both tables are identical.
//       More specifically, this is a naive join in the simplest scenario.
Result<shared_ptr<Table>> ExtendTable(
    const shared_ptr<const Table>& base_table,
    const shared_ptr<const Table>& ext_table) {
    // merge schema metadata from both tables
    auto ext_schema = ext_table->schema();
    auto base_schema = base_table->schema();

    shared_ptr<const ::arrow::KeyValueMetadata> base_schema_meta;
    shared_ptr<const ::arrow::KeyValueMetadata> combined_meta;
    if (base_schema->HasMetadata() && ext_schema->HasMetadata()) {
        base_schema_meta = base_schema->metadata()->Copy();
        combined_meta = base_schema_meta->Merge(*ext_schema->metadata());
    }

    // Concatenate schema fields
    ::arrow::FieldVector combined_fields{base_schema->fields()};
    combined_fields.insert(combined_fields.end()
                           // do this if we assume the first col is identical
                           // ,++ext_schema->fields().cbegin()
                           ,
                           ext_schema->fields().cbegin(),
                           ext_schema->fields().cend());

    // Concatenate table columns
    ::arrow::ChunkedArrayVector combined_cols{base_table->columns()};
    combined_cols.insert(combined_cols.end()
                         // do this if we assume the first col is identical
                         // ,++ext_table->columns().cbegin()
                         ,
                         ext_table->columns().cbegin(),
                         ext_table->columns().cend());

    // Construct new table from concatenated schema fields and table columns
    return Table::Make(::arrow::schema(combined_fields, combined_meta),
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

    ::arrow::ChunkedArrayVector combined_cols;
    ::arrow::SchemaBuilder schema_builder;
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

    std::shared_ptr<::arrow::Schema> schema;
    ARROW_ASSIGN_OR_RAISE(schema, schema_builder.Finish());

    return Table::Make(schema, combined_cols);
}

// see arrow's cpp/src/arrow/json/reader_test.cc
inline static Status MakeStream(std::string_view src_str,
                                shared_ptr<::arrow::io::InputStream>* out) {
    auto src = make_shared<::arrow::Buffer>(src_str);
    *out = make_shared<::arrow::io::BufferReader>(src);
    return Status::OK();
}

Result<shared_ptr<Table>> ReadToTable(
    string json, const ::arrow::json::ReadOptions& read_options,
    const ::arrow::json::ParseOptions& parse_options) {
    shared_ptr<::arrow::io::InputStream> input;
    RETURN_NOT_OK(MakeStream(json, &input));
    ARROW_ASSIGN_OR_RAISE(auto reader, ::arrow::json::TableReader::Make(
                                           ::arrow::default_memory_pool(), input,
                                           read_options, parse_options));
    return reader->Read();
}

}  // namespace YABTE::Utilities::Arrow
