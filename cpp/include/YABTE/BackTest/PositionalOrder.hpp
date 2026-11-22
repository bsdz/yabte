#pragma once

#include "YABTE/BackTest/Order.hpp"

namespace YABTE::BackTest {

enum PositionalOrderCheckType { POS_TQ_DIFFER = 1, ZERO_POS = 2 };

class PositionalOrder : public SimpleOrder {
   public:
    PositionalOrder(
        const string &asset_name, const double &size,
        const OrderSizeType &size_type = OrderSizeType::QUANTITY,
        const PositionalOrderCheckType &check_type = POS_TQ_DIFFER,
        const optional<string> &book_name = nullopt,
        const optional<string> &label = nullopt, const int priority = 0,
        const optional<string> &key = nullopt, bool close_first = true);

    shared_ptr<Order> clone() const override;

    PositionalOrderCheckType check_type_;
    // true: close the current position, then open the target (two trades);
    // false: trade only the difference (one trade)
    bool close_first_;

    void apply(const Timestamp &ts, const DayData &day_data,
               const AssetMap &asset_map) override;

    void apply(const Timestamp &ts, const std::shared_ptr<arrow::Table>& data, int64_t idx, const AssetMap &asset_map) override;

   private:
    // checks the position against the target and books the trades
    void trade_to_target(const Timestamp &ts, double trade_quantity,
                         double trade_price, const AssetMap &asset_map);
};
}  // namespace YABTE::BackTest
