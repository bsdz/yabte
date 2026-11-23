#include "YABTE/BackTest/Order.hpp"

#include <glog/logging.h>

#include <map>
#include <stdexcept>
#include <utility>

using std::runtime_error, std::make_shared, std::map, std::pair;

namespace YABTE::BackTest {

Order::Order(const optional<string>& book_name, const optional<string>& label,
             const int priority, const optional<string>& key)
    : status_(OrderStatus::OPEN),
      book_name_(book_name),
      label_(label),
      priority_(priority),
      key_(key) {}

void Order::_book_trades(const vector<shared_ptr<Trade>> trades,
                         const AssetMap& asset_map, const arrow::Table& data,
                         int64_t idx) {
    // test then book trades, do any post complete tasks
    if (this->book_->test_trades(trades)) {
        TransactionVector transactions(trades.begin(), trades.end());
        // each asset's cost, charged on the net quantity the order trades in it
        // (as the Python Order._costs), so closing and reopening pays only on
        // the difference
        vector<string> asset_names;
        map<string, pair<double, double>> net_and_price;
        for (const auto& trade : trades) {
            auto [it, added] = net_and_price.try_emplace(
                trade->asset_name_, 0., trade->price_);
            if (added) asset_names.push_back(trade->asset_name_);
            it->second.first += trade->quantity_;
        }
        for (const auto& name : asset_names) {
            const auto& asset = asset_map.at(name);
            auto quantity = asset->round_quantity(net_and_price[name].first);
            auto price = net_and_price[name].second;
            const auto& ts = trades.front()->ts_;
            // one Cost per source: the fixed cost, then the cost from data
            if (auto cost = asset->transaction_cost(quantity, price); cost != 0) {
                transactions.push_back(make_shared<Cost>(
                    ts, cost, name, this->label_, quantity, asset->cost_type_));
            }
            if (auto cost = asset->data_cost(quantity, price, data, idx, ts);
                cost != 0) {
                transactions.push_back(make_shared<Cost>(
                    ts, cost, name, this->label_, quantity,
                    asset->cost_field_type_, asset->cost_field_));
            }
        }
        this->book_->add_transactions(transactions);
        // Python sums Decimal quantities exactly; doubles drift off the asset's
        // quantity grid, so snap back to it (a flat position must be exactly 0)
        for (const auto& trade : trades) {
            auto& position = this->book_->positions_[trade->asset_name_];
            position = asset_map.at(trade->asset_name_)->round_quantity(position);
        }
        this->status_ = OrderStatus::COMPLETE;
        this->post_complete(trades);
    } else {
        this->status_ = OrderStatus::MANDATE_FAILED;
    }
}

void Order::post_complete(const vector<shared_ptr<Trade>> trades) {}

void Order::apply(const Timestamp &ts, const std::shared_ptr<arrow::Table>& data, int64_t idx, const AssetMap &asset_map) {
    auto day_data = data->Slice(idx, 1);
    this->apply(ts, *day_data, asset_map);
}

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
    else if (this->size_type_ == OrderSizeType::BOOK_PERCENT) {
        // a percentage of the book's cash, as the Python engine
        return {asset->round_quantity(this->book_->cash_ * this->size_ / 100 /
                                      trade_price),
                trade_price};
    }     else
        throw runtime_error("Unsupported size type");
}

tuple<double, double> SimpleOrder::_calc_quantity_price(
    const std::shared_ptr<arrow::Table>& data, int64_t idx,
    const AssetMap &asset_map) const {
    auto asset = asset_map.at(this->asset_name_);
    // Use new efficient API on Asset
    auto trade_price =
        asset->intraday_traded_price(data, idx, this->size_);

    if (this->size_type_ == OrderSizeType::QUANTITY)
        return {asset->round_quantity(this->size_), trade_price};
    else if (this->size_type_ == OrderSizeType::NOTIONAL)
        return {asset->round_quantity(this->size_ / trade_price), trade_price};
    else if (this->size_type_ == OrderSizeType::BOOK_PERCENT) {
        // a percentage of the book's cash, as the Python engine
        return {asset->round_quantity(this->book_->cash_ * this->size_ / 100 /
                                      trade_price),
                trade_price};
    } else
        throw runtime_error("Unsupported size type");
}

optional<OrderStatus> SimpleOrder::pre_execute_check(const Timestamp& ts,
                                                     double trade_price) const {
    return nullopt;
}

void SimpleOrder::apply(const Timestamp& ts, const DayData& day_data,
                        const AssetMap& asset_map) {
    // DLOG(INFO) << "SimpleOrder::apply()";
    if (!this->book_) {
        throw runtime_error("Book not found");
    }

    auto [trade_quantity, trade_price] =
        this->_calc_quantity_price(day_data, asset_map);
    require_price(trade_price, this->asset_name_, ts, "traded");

    auto new_status = this->pre_execute_check(ts, trade_price);
    if (new_status) {
        this->status_ = *new_status;
        return;
    }

    vector<shared_ptr<Trade>> trades;
    trades.push_back(make_shared<Trade>(ts, trade_quantity, trade_price,
                                        this->asset_name_, this->label_));
    this->_book_trades(trades, asset_map, day_data, 0);
}

void SimpleOrder::apply(const Timestamp &ts, const std::shared_ptr<arrow::Table>& data, int64_t idx, const AssetMap &asset_map) {
    if (!this->book_) {
        throw runtime_error("Book not found");
    }

    auto [trade_quantity, trade_price] =
        this->_calc_quantity_price(data, idx, asset_map);
    require_price(trade_price, this->asset_name_, ts, "traded");

    auto new_status = this->pre_execute_check(ts, trade_price);
    if (new_status) {
        this->status_ = *new_status;
        return;
    }

    vector<shared_ptr<Trade>> trades;
    trades.push_back(make_shared<Trade>(ts, trade_quantity, trade_price,
                                        this->asset_name_, this->label_));
    this->_book_trades(trades, asset_map, *data, idx);
}


shared_ptr<Order> SimpleOrder::clone() const {
    return make_shared<SimpleOrder>(*this);
}

}  // namespace YABTE::BackTest