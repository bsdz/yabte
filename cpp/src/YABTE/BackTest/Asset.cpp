#include "YABTE/BackTest/Asset.hpp"

#include <arrow/table.h>

#include <iostream>
#include <regex>
#include <stdexcept>

using std::dynamic_pointer_cast, std::make_shared;

namespace YABTE::BackTest {

Asset::Asset(const string &name, const string &denom,
             const optional<int> &price_round_dp,
             const optional<int> &quantity_round_dp,
             const optional<string> &data_label)
    : name_(name),
      denom_(denom),
      price_round_dp_(price_round_dp),
      quantity_round_dp_(quantity_round_dp) {
    if (data_label.has_value()) {
        this->data_label_ = data_label.value();
    } else {
        this->data_label_ = name;
    }
}

double Asset::round_quantity(const double &quantity) const {
    return round_dp(quantity, this->quantity_round_dp_);
}

vector<string> Asset::_get_fields(const AssetDataFieldInfo &field_info) const {
    vector<string> res;
    for (auto const &[fn, fi] : this->data_fields())
        if (fi & field_info) res.push_back(fn);
    return res;
}

arrow::Status _FilterTable(const DayData &day_data, vector<int> &indices,
                           vector<string> &new_names,
                           shared_ptr<DayData> &day_data_filt) {
    ARROW_ASSIGN_OR_RAISE(auto x, day_data.SelectColumns(indices));
    ARROW_ASSIGN_OR_RAISE(auto y, x->RenameColumns(new_names));

    day_data_filt.swap(y);
    y.reset();

    return arrow::Status::OK();
}

shared_ptr<DayData> Asset::_filter_data(const DayData &day_data) const {
    // R"(^\('(\S+)', '(\S+)'\)$)" matches python tuple string representation
    // e.g. "('GOOG', 'Close')" -> capture group 1: "GOOG", capture group 2:
    // "Close"
    const std::regex field_re(R"(^\('(\S+)', '(\S+)'\)$)");

    vector<int> indices;
    vector<string> new_names;

    // Check if we are using MultiIndex columns (tuple strings) or flat columns
    // (Dot notation) The previous debug output shows columns like "GOOG.Open",
    // "GOOG.High", etc. This suggests the column names are NOT tuple strings
    // but "Asset.Field".

    bool is_tuple_format = false;
    if (day_data.num_columns() > 0) {
        std::string first_col = day_data.ColumnNames()[0];
        if (first_col.find("('") == 0) {
            is_tuple_format = true;
        }
    }

    if (is_tuple_format) {
        for (auto const &cn : day_data.ColumnNames()) {
            const vector<std::smatch> matches{
                std::sregex_iterator{cn.begin(), cn.end(), field_re},
                std::sregex_iterator{}};

            if (matches.size() > 0 && matches[0].str(1) == this->data_label_) {
                indices.push_back(day_data.schema()->GetFieldIndex(cn));
                new_names.push_back(matches[0].str(2));
            }
        }
    } else {
        // Fallback to "Asset.Field" format
        std::string prefix = this->data_label_ + ".";
        for (auto const &cn : day_data.ColumnNames()) {
            if (cn.find(prefix) == 0) {
                indices.push_back(day_data.schema()->GetFieldIndex(cn));
                new_names.push_back(cn.substr(prefix.length()));
            }
        }
    }

    if (indices.empty()) {
        std::cerr << "WARNING: No columns found for asset " << this->name_
                  << " (label: " << this->data_label_ << ")" << std::endl;
        std::cerr << "Available columns: ";
        for (const auto &cn : day_data.ColumnNames()) std::cerr << cn << ", ";
        std::cerr << std::endl;
    }

    auto day_data_filt = day_data.Slice(0, 1);
    auto st = _FilterTable(day_data, indices, new_names, day_data_filt);
    if (!st.ok()) {
        throw std::runtime_error("Error: " + st.ToString());
    }
    return day_data_filt;
}

double Asset::intraday_traded_price(const std::shared_ptr<arrow::Table>& data,
                                    int64_t idx,
                                    const optional<double> size) const {
    // Default implementation for backward compatibility (e.g. Python assets)
    // Slices the table and calls the old method.
    // Note: The old method expects *filtered* data if called from SimpleOrder,
    // but here we are just adapting the signature.
    // However, subclasses like PyAsset override the old method which expects
    // filtered data usually passed from outside.
    // If we call the old method here with raw data, it might fail.
    // But wait, the StrategyRunner loop calls Order::apply with raw sliced data.
    // Order::apply calls _filter_data then asset->intraday_traded_price.
    
    // If we change StrategyRunner to call this new method directly (or via Order),
    // we are skipping _filter_data.
    // So this default implementation MUST perform _filter_data logic if the underlying
    // old method expects it.
    
    // Actually, let's assume this method is only called by updated Orders/Runners.
    // If a subclass (like PyAsset) hasn't overridden this, we must ensure
    // we provide what the old method expects.
    
    auto day_data = data->Slice(idx, 1);
    auto filtered = this->_filter_data(*day_data);
    return this->intraday_traded_price(*filtered, size);
}

double Asset::end_of_day_price(const std::shared_ptr<arrow::Table>& data,
                               int64_t idx) const {
    auto day_data = data->Slice(idx, 1);
    auto filtered = this->_filter_data(*day_data);
    return this->end_of_day_price(*filtered);
}

OHLCAsset::OHLCAsset(const string &name, const string &denom,
                     const optional<int> &price_round_dp,
                     const optional<int> &quantity_round_dp,
                     const optional<string> &data_label)
    : Asset(name, denom, price_round_dp, quantity_round_dp, data_label) {}

shared_ptr<Asset> OHLCAsset::clone() const {
    return make_shared<OHLCAsset>(*this);
};

double OHLCAsset::intraday_traded_price(const DayData &asset_day_data,
                                        const optional<double> size) const {
    auto s_low = asset_day_data.GetColumnByName("Low");
    if (!s_low) {
        std::cerr << "ERROR: 'Low' column missing for " << this->name_
                  << std::endl;
        throw std::runtime_error("Low column missing");
    }

    auto s_high = asset_day_data.GetColumnByName("High");
    if (!s_high) {
        std::cerr << "ERROR: 'High' column missing for " << this->name_
                  << std::endl;
        throw std::runtime_error("High column missing");
    }

    auto st_s_low = s_low->GetScalar(0);
    auto st_s_high = s_high->GetScalar(0);

    if (st_s_low.ok() && st_s_high.ok()) {
        auto low_scalar = st_s_low.ValueOrDie();
        auto high_scalar = st_s_high.ValueOrDie();

        if (low_scalar && high_scalar) {
            auto low =
                dynamic_pointer_cast<arrow::DoubleScalar>(low_scalar)->value;
            auto high =
                dynamic_pointer_cast<arrow::DoubleScalar>(high_scalar)->value;

            if (!std::isnan(low) && !std::isnan(high)) {
                return round_dp((low + high) / 2, this->price_round_dp_);
            }
        }
    }

    auto s_close = asset_day_data.GetColumnByName("Close");
    auto st_s_close = s_close->GetScalar(0);

    if (st_s_close.ok()) {
        auto close_scalar = st_s_close.ValueOrDie();
        if (close_scalar) {
            auto close =
                dynamic_pointer_cast<arrow::DoubleScalar>(close_scalar)->value;
            return round_dp(close, this->price_round_dp_);
        }
    }

    throw std::runtime_error("Unable to determine intraday traded price");
}

double OHLCAsset::end_of_day_price(const DayData &asset_day_data) const {
    auto s_close = asset_day_data.GetColumnByName("Close");
    auto st_s_close = s_close->GetScalar(0);

    if (st_s_close.ok()) {
        auto close_scalar = st_s_close.ValueOrDie();
        if (close_scalar) {
            auto close =
                dynamic_pointer_cast<arrow::DoubleScalar>(close_scalar)->value;
            return round_dp(close, this->price_round_dp_);
        }
    }

    throw std::runtime_error("Unable to determine end of day price");
}

vector<tuple<string, AssetDataFieldInfo>> OHLCAsset::data_fields() const {
    vector<tuple<string, AssetDataFieldInfo>> v;
    v.emplace_back("High", AssetDataFieldInfo::AVAILABLE_AT_CLOSE);
    v.emplace_back("Low", AssetDataFieldInfo::AVAILABLE_AT_CLOSE);
    v.emplace_back("Open", (AssetDataFieldInfo::AVAILABLE_AT_CLOSE |
                            AssetDataFieldInfo::AVAILABLE_AT_OPEN));
    v.emplace_back("Close", (AssetDataFieldInfo::AVAILABLE_AT_CLOSE |
                             AssetDataFieldInfo::REQUIRED));
    v.emplace_back("Volume", AssetDataFieldInfo::AVAILABLE_AT_CLOSE);
    return v;
}

// Helper to get value from ChunkedArray at index
inline std::optional<double> GetValue(const std::shared_ptr<arrow::ChunkedArray>& arr, int64_t i) {
    if (!arr) return std::nullopt;
    // Fast path for single chunk
    if (arr->num_chunks() == 1) {
        auto carray = std::static_pointer_cast<arrow::DoubleArray>(arr->chunk(0));
        if (carray->IsNull(i)) return std::nullopt;
        return carray->Value(i);
    }
    // General path
    // TODO: Optimize by caching last chunk index/offset if needed
    int64_t offset = 0;
    for (const auto& chunk : arr->chunks()) {
        if (i < offset + chunk->length()) {
             auto carray = std::static_pointer_cast<arrow::DoubleArray>(chunk);
             auto local_idx = i - offset;
             if (carray->IsNull(local_idx)) return std::nullopt;
             return carray->Value(local_idx);
        }
        offset += chunk->length();
    }
    return std::nullopt;
}

double OHLCAsset::intraday_traded_price(const std::shared_ptr<arrow::Table>& data,
                                        int64_t idx,
                                        const optional<double> size) const {
    if (!indices_initialized_) {
        auto schema = data->schema();
        auto find_col = [&](const std::string& field) -> int {
            // Try tuple format
            std::string name = "('" + this->data_label_ + "', '" + field + "')";
            int i = schema->GetFieldIndex(name);
            if (i != -1) return i;
            // Try dot format
            name = this->data_label_ + "." + field;
            return schema->GetFieldIndex(name);
        };
        
        idx_low_ = find_col("Low");
        idx_high_ = find_col("High");
        idx_close_ = find_col("Close");
        // idx_open_ = find_col("Open"); 
        indices_initialized_ = true;
    }

    std::optional<double> low, high;
    
    if (idx_low_ != -1) low = GetValue(data->column(idx_low_), idx);
    if (idx_high_ != -1) high = GetValue(data->column(idx_high_), idx);

    if (low.has_value() && high.has_value() && !std::isnan(low.value()) && !std::isnan(high.value())) {
        return round_dp((low.value() + high.value()) / 2.0, this->price_round_dp_);
    }

    if (idx_close_ != -1) {
        auto close = GetValue(data->column(idx_close_), idx);
        if (close.has_value()) {
            return round_dp(close.value(), this->price_round_dp_);
        }
    }

    throw std::runtime_error("Unable to determine intraday traded price for " + this->name_);
}

double OHLCAsset::end_of_day_price(const std::shared_ptr<arrow::Table>& data,
                                   int64_t idx) const {
    if (!indices_initialized_) {
         auto schema = data->schema();
         auto find_col = [&](const std::string& field) -> int {
            std::string name = "('" + this->data_label_ + "', '" + field + "')";
            int i = schema->GetFieldIndex(name);
            if (i != -1) return i;
            name = this->data_label_ + "." + field;
            return schema->GetFieldIndex(name);
        };
        idx_low_ = find_col("Low");
        idx_high_ = find_col("High");
        idx_close_ = find_col("Close");
        indices_initialized_ = true;
    }

    if (idx_close_ != -1) {
        auto close = GetValue(data->column(idx_close_), idx);
        if (close.has_value()) {
             return round_dp(close.value(), this->price_round_dp_);
        }
    }

    throw std::runtime_error("Unable to determine end of day price for " + this->name_);
}

}  // namespace YABTE::BackTest
