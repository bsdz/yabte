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
        const optional<string> &key = nullopt);

    shared_ptr<Order> clone() const override;

    PositionalOrderCheckType check_type_;

    void apply(const Timestamp &ts, const DayData &day_data,
               const AssetMap &asset_map) override;
};

}  // namespace YABTE::BackTest
