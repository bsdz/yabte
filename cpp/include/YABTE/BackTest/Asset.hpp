#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "YABTE/BackTest/common.hpp"

using std::map, std::nullopt, std::optional, std::shared_ptr, std::string,
    std::tuple, std::vector;

namespace YABTE::BackTest {

enum AssetDataFieldInfo {
    AVAILABLE_AT_CLOSE = 1,
    AVAILABLE_AT_OPEN = 2,
    REQUIRED = 4
};

inline AssetDataFieldInfo operator|(AssetDataFieldInfo a,
                                    AssetDataFieldInfo b) {
    return static_cast<AssetDataFieldInfo>(static_cast<int>(a) |
                                           static_cast<int>(b));
}

class Asset {
   public:
    virtual ~Asset() = default;
    virtual shared_ptr<Asset> clone() const = 0;

    double round_quantity(const double &quantity) const;
    virtual double intraday_traded_price(
        const DayData &asset_day_data,
        const optional<double> size = nullopt) const = 0;
    virtual double end_of_day_price(const DayData &asset_day_data) const = 0;

    virtual vector<tuple<string, AssetDataFieldInfo>> data_fields() const = 0;
    vector<string> _get_fields(const AssetDataFieldInfo &field_info) const;
    shared_ptr<DayData> _filter_data(const DayData &day_data) const;

    string name_;
    string denom_;
    int price_round_dp_;
    int quantity_round_dp_;
    string data_label_;

   protected:
    Asset(const string &name, const string &denom, const int price_round_dp = 2,
          const int quantity_round_dp = 2,
          const optional<string> &data_label = nullopt);
};

using AssetMap = map<string, shared_ptr<Asset>>;
using AssetVector = vector<shared_ptr<Asset>>;

class OHLCAsset : public Asset {
   public:
    OHLCAsset(const string &name, const string &denom,
              const int price_round_dp = 2, const int quantity_round_dp = 2,
              const optional<string> &data_label = nullopt);

    shared_ptr<Asset> clone() const override;

    double intraday_traded_price(
        const DayData &asset_day_data,
        const optional<double> size = nullopt) const override;
    double end_of_day_price(const DayData &asset_day_data) const override;
    vector<tuple<string, AssetDataFieldInfo>> data_fields() const override;
};
}  // namespace YABTE::BackTest