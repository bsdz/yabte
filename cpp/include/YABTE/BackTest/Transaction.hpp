#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "YABTE/BackTest/common.hpp"

using std::string, std::string_literals::operator""s, std::optional,
    std::shared_ptr, std::vector;

namespace YABTE::BackTest {
class Transaction {
   public:
    Transaction(const Timestamp &ts, const double &total = 0.,
                const string &desc = ""s);
    // to ensure polymorphic
    virtual ~Transaction() = default;
    virtual shared_ptr<Transaction> clone() const;

    Timestamp ts_;
    double total_;
    string desc_;
};

using TransactionVector = vector<shared_ptr<Transaction>>;

class CashTransaction : public Transaction {
   public:
    CashTransaction(const Timestamp &ts, const double &total,
                    const string &desc = ""s);
    virtual shared_ptr<Transaction> clone() const;
};

class Trade : public Transaction {
   public:
    Trade(const Timestamp &ts, const double &quantity, const double &price,
          const string &asset_name, const optional<string> &order_label = ""s);
    virtual shared_ptr<Transaction> clone() const;

    double quantity_;
    double price_;
    string asset_name_;
    optional<string> order_label_;
};

}  // namespace YABTE::BackTest