#pragma once

#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "YABTE/BackTest/Asset.hpp"
#include "YABTE/BackTest/Book.hpp"
#include "YABTE/BackTest/Transaction.hpp"
#include "YABTE/BackTest/common.hpp"

using std::deque, std::nullopt, std::optional, std::shared_ptr, std::string,
    std::vector, std::tuple;

namespace YABTE::BackTest {

enum OrderStatus {
    MANDATE_FAILED = 1,
    CANCELLED = 2,
    OPEN = 3,
    COMPLETE = 4,
    REPLACED = 5
};

enum OrderSizeType {
    QUANTITY = 1,
    NOTIONAL = 2,
    BOOK_PERCENT = 3,
};

class Order {
   public:
    Order(const optional<string> &book_name = nullopt,
          const optional<string> &label = nullopt, const int priority = 0,
          const optional<string> &key = nullopt);
    virtual ~Order() = default;
    virtual shared_ptr<Order> clone() const = 0;

    void _book_trades(const vector<shared_ptr<Trade>> trades);

    virtual void post_complete(const vector<shared_ptr<Trade>> trades);

    virtual void apply(const Timestamp &ts, const DayData &day_data,
                       const AssetMap &asset_map) = 0;

    OrderStatus status_;
    optional<string> book_name_;
    shared_ptr<Book> book_;
    vector<shared_ptr<Order>> suborders_;
    optional<string> label_;
    int priority_;
    optional<string> key_;
};

using OrderDeque = deque<shared_ptr<Order>>;
using OrderVector = vector<shared_ptr<Order>>;

class SimpleOrder : public Order {
   public:
    SimpleOrder(const string &asset_name, const double &size,
                const OrderSizeType &size_type = OrderSizeType::QUANTITY,
                const optional<string> &book_name = nullopt,
                const optional<string> &label = nullopt, const int priority = 0,
                const optional<string> &key = nullopt);
    shared_ptr<Order> clone() const override;

    string asset_name_;
    double size_;
    OrderSizeType size_type_;

    optional<OrderStatus> pre_execute_check(const Timestamp &ts,
                                            const double trade_price) const;
    tuple<double, double> _calc_quantity_price(const DayData &day_data,
                                               const AssetMap &asset_map) const;

    void apply(const Timestamp &ts, const DayData &day_data,
               const AssetMap &asset_map) override;
};
}  // namespace YABTE::BackTest