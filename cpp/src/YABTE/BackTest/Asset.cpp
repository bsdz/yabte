#include "YABTE/BackTest/Asset.hpp"

#include <arrow/table.h>

#include <iostream>
#include <regex>
#include <stdexcept>

using std::dynamic_pointer_cast, std::make_shared;

namespace YABTE::BackTest {

Asset::Asset(const string &name, const string &denom, const int price_round_dp,
             const int quantity_round_dp, const optional<string> &data_label)
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
    return round_n_digits(quantity, this->quantity_round_dp_);
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
    // e.g. "('GOOG', 'Close')" -> capture group 1: "GOOG", capture group 2: "Close"
    const std::regex field_re(R"(^\('(\S+)', '(\S+)'\)$)");

    vector<int> indices;
    vector<string> new_names;

    // Check if we are using MultiIndex columns (tuple strings) or flat columns (Dot notation)
    // The previous debug output shows columns like "GOOG.Open", "GOOG.High", etc.
    // This suggests the column names are NOT tuple strings but "Asset.Field".
    
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
        std::cerr << "WARNING: No columns found for asset " << this->name_ << " (label: " << this->data_label_ << ")" << std::endl;
        std::cerr << "Available columns: ";
        for (const auto& cn : day_data.ColumnNames()) std::cerr << cn << ", ";
        std::cerr << std::endl;
    }

    auto day_data_filt = day_data.Slice(0, 1);
    auto st = _FilterTable(day_data, indices, new_names, day_data_filt);
    if (!st.ok()) {
        throw std::runtime_error("Error: " + st.ToString());
    }
    return day_data_filt;
}

OHLCAsset::OHLCAsset(const string &name, const string &denom,
                     const int price_round_dp, const int quantity_round_dp,
                     const optional<string> &data_label)
    : Asset(name, denom, price_round_dp, quantity_round_dp, data_label) {}

shared_ptr<Asset> OHLCAsset::clone() const {
    return make_shared<OHLCAsset>(*this);
};

double OHLCAsset::intraday_traded_price(const DayData &asset_day_data,
                                        const optional<double> size) const {
    
    auto s_low = asset_day_data.GetColumnByName("Low");
    if (!s_low) {
         std::cerr << "ERROR: 'Low' column missing for " << this->name_ << std::endl;
         throw std::runtime_error("Low column missing");
    }

    auto s_high = asset_day_data.GetColumnByName("High");
    if (!s_high) {
        std::cerr << "ERROR: 'High' column missing for " << this->name_ << std::endl;
        throw std::runtime_error("High column missing");
    }

    auto st_s_low = s_low->GetScalar(0);
    auto st_s_high = s_high->GetScalar(0);

    if (st_s_low.ok() && st_s_high.ok()) {
        auto low_scalar = st_s_low.ValueOrDie();
        auto high_scalar = st_s_high.ValueOrDie();

        if (low_scalar && high_scalar) {
             auto low = dynamic_pointer_cast<arrow::DoubleScalar>(low_scalar)->value;
             auto high = dynamic_pointer_cast<arrow::DoubleScalar>(high_scalar)->value;

             if (!std::isnan(low) && !std::isnan(high)) {
                 return round_n_digits((low + high) / 2, this->price_round_dp_);
             }
        }
    }

    auto s_close = asset_day_data.GetColumnByName("Close");
    auto st_s_close = s_close->GetScalar(0);

    if (st_s_close.ok()) {
        auto close_scalar = st_s_close.ValueOrDie();
        if (close_scalar) {
             auto close = dynamic_pointer_cast<arrow::DoubleScalar>(close_scalar)->value;
             return round_n_digits(close, this->price_round_dp_);
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
             auto close = dynamic_pointer_cast<arrow::DoubleScalar>(close_scalar)->value;
             return round_n_digits(close, this->price_round_dp_);
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

}  // namespace YABTE::BackTest
