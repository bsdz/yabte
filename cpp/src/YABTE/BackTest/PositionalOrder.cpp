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
                                 const optional<string>& key,
                                 bool close_first)
    : SimpleOrder(asset_name, size, size_type, book_name, label, priority, key),
      check_type_(check_type),
      close_first_(close_first) {}

void PositionalOrder::apply(const Timestamp& ts, const DayData& day_data,
                            const AssetMap& asset_map) {
    if (!this->book_) {
        throw runtime_error("Cannot apply order without book instance");
    }

    auto [trade_quantity, trade_price] =
        this->_calc_quantity_price(day_data, asset_map);
    require_price(trade_price, this->asset_name_, ts, "traded");

    auto new_status = this->pre_execute_check(ts, trade_price);
    if (new_status) {
        this->status_ = *new_status;
        return;
    }

    this->trade_to_target(ts, trade_quantity, trade_price, asset_map);
}

void PositionalOrder::apply(const Timestamp& ts, const std::shared_ptr<arrow::Table>& data, int64_t idx,
                            const AssetMap& asset_map) {
    if (!this->book_) {
        throw runtime_error("Cannot apply order without book instance");
    }

    auto [trade_quantity, trade_price] =
        this->_calc_quantity_price(data, idx, asset_map);
    require_price(trade_price, this->asset_name_, ts, "traded");

    auto new_status = this->pre_execute_check(ts, trade_price);
    if (new_status) {
        this->status_ = *new_status;
        return;
    }

    this->trade_to_target(ts, trade_quantity, trade_price, asset_map);
}

void PositionalOrder::trade_to_target(const Timestamp& ts, double trade_quantity,
                                      double trade_price,
                                      const AssetMap& asset_map) {
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

    if (needs_trades && !this->close_first_) {
        // round, as doubles don't subtract exactly
        auto difference = asset_map.at(this->asset_name_)
                              ->round_quantity(trade_quantity - current_position);
        if (difference != 0) {
            trades.push_back(make_shared<Trade>(ts, difference, trade_price,
                                                this->asset_name_,
                                                this->label_));
        }
    } else if (needs_trades) {
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

    this->_book_trades(trades, asset_map);
}

shared_ptr<Order> PositionalOrder::clone() const {
    return make_shared<PositionalOrder>(*this);
}

}  // namespace YABTE::BackTest
