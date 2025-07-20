#include "YABTE/BackTest/Order.hpp"

#include <glog/logging.h>

#include <stdexcept>

using std::runtime_error, std::make_shared;

namespace YABTE::BackTest {

Order::Order(const optional<string>& book_name, const optional<string>& label,
             const int priority, const optional<string>& key)
    : status_(OrderStatus::OPEN),
      book_name_(book_name),
      label_(label),
      priority_(priority),
      key_(key) {}

void Order::_book_trades(const vector<shared_ptr<Trade>> trades) {
    // test then book trades, do any post complete tasks
    if (this->book_->test_trades(trades)) {
        TransactionVector transactions(trades.begin(), trades.end());
        this->book_->add_transactions(transactions);
        this->status_ = OrderStatus::COMPLETE;
        this->post_complete(trades);
    } else {
        this->status_ = OrderStatus::MANDATE_FAILED;
    }
}

void Order::post_complete(const vector<shared_ptr<Trade>> trades) {}

SimpleOrder::SimpleOrder(const string& asset_name, const double& size,
                         const OrderSizeType& size_type,
                         const optional<string>& book_name,
                         const optional<string>& label, const int priority,
                         const optional<string>& key)
    : Order(book_name, label, priority, key),
      asset_name_(asset_name),
      size_(size),
      size_type_(size_type) {}

tuple<double, double> SimpleOrder::_calc_quantity_price(
    const DayData& day_data, const AssetMap& asset_map) const {
    auto asset = asset_map.at(this->asset_name_);
    auto asset_day_data = asset->_filter_data(day_data);
    auto trade_price =
        asset->intraday_traded_price(*asset_day_data, this->size_);

    if (this->size_type_ == OrderSizeType::QUANTITY)
        return {asset->round_quantity(this->size_), trade_price};
    else if (this->size_type_ == OrderSizeType::NOTIONAL)
        return {asset->round_quantity(this->size_ / trade_price), trade_price};
    else if (this->size_type_ == OrderSizeType::BOOK_PERCENT)
        return {asset->round_quantity(this->book_->cash_ * this->size_ / 100 /
                                      trade_price),
                trade_price};
    else
        throw runtime_error("Unsupported size type");
}

optional<OrderStatus> SimpleOrder::pre_execute_check(const Timestamp& ts,
                                                     double trade_price) const {
    return nullopt;
}

void SimpleOrder::apply(const Timestamp& ts, const DayData& day_data,
                        const AssetMap& asset_map) {
    DLOG(INFO) << "SimpleOrder::apply()";
    if (!this->book_) {
        throw runtime_error("Book not found");
    }

    auto [trade_quantity, trade_price] =
        this->_calc_quantity_price(day_data, asset_map);

    auto new_status = this->pre_execute_check(ts, trade_price);
    if (new_status) {
        this->status_ = *new_status;
        return;
    }

    vector<shared_ptr<Trade>> trades;
    trades.push_back(make_shared<Trade>(ts, trade_quantity, trade_price,
                                        this->asset_name_, this->label_));
    this->_book_trades(trades);
}

shared_ptr<Order> SimpleOrder::clone() const {
    return make_shared<SimpleOrder>(*this);
}

}  // namespace YABTE::BackTest