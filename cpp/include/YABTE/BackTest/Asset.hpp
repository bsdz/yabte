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

    // The fixed cost of one order trading a net `quantity` at `price`; see the
    // Python Asset.transaction_cost.
    double transaction_cost(double quantity, double price) const;

    // The cost read from `cost_field_` in row `idx` of `data` (flattened
    // "label.field" columns); see the Python Asset.data_cost. Throws
    // MissingPriceError for a missing value and invalid_argument for a negative
    // one, but only when `quantity` isn't zero.
    double data_cost(double quantity, double price, const arrow::Table &data,
                     int64_t idx, const Timestamp &ts) const;
    
    // New efficient methods that take full table and index
    virtual double intraday_traded_price(
        const std::shared_ptr<arrow::Table>& data, int64_t idx,
        const optional<double> size = nullopt) const;
    virtual double end_of_day_price(
        const std::shared_ptr<arrow::Table>& data, int64_t idx) const;

    // Old methods (kept pure virtual for backward compatibility with existing subclasses)
    virtual double intraday_traded_price(
        const DayData &asset_day_data,
        const optional<double> size = nullopt) const = 0;
    virtual double end_of_day_price(const DayData &asset_day_data) const = 0;

    virtual vector<tuple<string, AssetDataFieldInfo>> data_fields() const = 0;
    vector<string> _get_fields(const AssetDataFieldInfo &field_info) const;
    shared_ptr<DayData> _filter_data(const DayData &day_data) const;

    string name_;
    string denom_;
    optional<int> price_round_dp_;
    optional<int> quantity_round_dp_;
    string data_label_;
    double cost_ = 0.;
    CostType cost_type_ = CostType::BPS;
    optional<string> cost_field_;
    CostType cost_field_type_ = CostType::PER_UNIT;

   protected:
    Asset(const string &name, const string &denom,
          const optional<int> &price_round_dp = 2,
          const optional<int> &quantity_round_dp = 2,
          const optional<string> &data_label = nullopt, double cost = 0.,
          CostType cost_type = CostType::BPS,
          const optional<string> &cost_field = nullopt,
          CostType cost_field_type = CostType::PER_UNIT);
};

using AssetMap = map<string, shared_ptr<Asset>>;
using AssetVector = vector<shared_ptr<Asset>>;

// The price OHLCAsset orders fill at; see the Python OHLCAsset.fill_price.
enum class FillPrice { MIDPOINT, OPEN, CLOSE };

class OHLCAsset : public Asset {
   public:
    OHLCAsset(const string &name, const string &denom,
              const optional<int> &price_round_dp = 2,
              const optional<int> &quantity_round_dp = 2,
              const optional<string> &data_label = nullopt,
              FillPrice fill_price = FillPrice::MIDPOINT, double cost = 0.,
              CostType cost_type = CostType::BPS,
              const optional<string> &cost_field = nullopt,
              CostType cost_field_type = CostType::PER_UNIT);

    FillPrice fill_price_;

    shared_ptr<Asset> clone() const override;

    double intraday_traded_price(
        const DayData &asset_day_data,
        const optional<double> size = nullopt) const override;
    double end_of_day_price(const DayData &asset_day_data) const override;
    
    // Efficient overrides
    double intraday_traded_price(
        const std::shared_ptr<arrow::Table>& data, int64_t idx,
        const optional<double> size = nullopt) const override;
    double end_of_day_price(
        const std::shared_ptr<arrow::Table>& data, int64_t idx) const override;

    vector<tuple<string, AssetDataFieldInfo>> data_fields() const override;

   private:
    mutable int idx_open_ = -1;
    mutable int idx_high_ = -1;
    mutable int idx_low_ = -1;
    mutable int idx_close_ = -1;
    mutable int idx_volume_ = -1;
    mutable bool indices_initialized_ = false;
};
}  // namespace YABTE::BackTest