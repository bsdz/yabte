#include "YABTE/BackTest/PositionalOrder.hpp"

#include <glog/logging.h>

#include <stdexcept>

using std::runtime_error, std::make_shared;

namespace YABTE::BackTest {

PositionalOrder::PositionalOrder(const string& asset_name, const double& size,
                                 const OrderSizeType& size_type,
                                 const PositionalOrderCheckType& check_type,
                                 const optional<string>& book_name,
                                 const optional<string>& label,
                                 const int priority,
                                 const optional<string>& key)
    : SimpleOrder(asset_name, size, size_type, book_name, label, priority, key),
      check_type_(check_type) {}

void PositionalOrder::apply(const Timestamp& ts, const DayData& day_data,
                            const AssetMap& asset_map) {
    if (!this->book_) {
        throw runtime_error("Cannot apply order without book instance");
    }

    auto [trade_quantity, trade_price] =
        this->_calc_quantity_price(day_data, asset_map);

    auto new_status = this->pre_execute_check(ts, trade_price);
    if (new_status) {
        this->status_ = *new_status;
        return;
    }

    auto current_position = this->book_->positions_[this->asset_name_];
    bool needs_trades = false;

    if (this->check_type_ == PositionalOrderCheckType::POS_TQ_DIFFER) {
        needs_trades = (current_position != trade_quantity);
    } else if (this->check_type_ == PositionalOrderCheckType::ZERO_POS) {
        needs_trades = (current_position == 0);
    } else {
        throw runtime_error("Unexpected check type");
    }

    vector<shared_ptr<Trade>> trades;

    if (needs_trades) {
        if (current_position != 0) {
            // close out existing position
            trades.push_back(make_shared<Trade>(
                ts, -current_position, trade_price, this->asset_name_,
                this->label_));
        }
        if (trade_quantity != 0) {
            trades.push_back(make_shared<Trade>(ts, trade_quantity, trade_price,
                                                this->asset_name_,
                                                this->label_));
        }
    }

    this->_book_trades(trades);
}

shared_ptr<Order> PositionalOrder::clone() const {
    return make_shared<PositionalOrder>(*this);
}

}  // namespace YABTE::BackTest
