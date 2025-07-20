#include "YABTE/BackTest/Transaction.hpp"

#include <memory>

using std::make_shared;

namespace YABTE::BackTest {

Transaction::Transaction(const Timestamp &ts, const double &total,
                         const string &desc)
    : ts_(ts), total_(total), desc_(desc){};

shared_ptr<Transaction> Transaction::clone() const {
    return make_shared<Transaction>(*this);
}

CashTransaction::CashTransaction(const Timestamp &ts, const double &total,
                                 const string &desc)
    : Transaction(ts, total, desc) {}

shared_ptr<Transaction> CashTransaction::clone() const {
    return make_shared<Transaction>(*this);
}

Trade::Trade(const Timestamp &ts, const double &quantity, const double &price,
             const string &asset_name, const optional<string> &order_label)
    : Transaction(ts),
      quantity_(quantity),
      price_(price),
      asset_name_(asset_name),
      order_label_(order_label) {
    this->total_ = -quantity * price;
    this->desc_ = (quantity < 0 ? "sell "s : "buy "s) + asset_name;
}
shared_ptr<Transaction> Trade::clone() const {
    return make_shared<Transaction>(*this);
}

}  // namespace YABTE::BackTest
