#pragma once

#include <arrow/table.h>

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "YABTE/BackTest/Asset.hpp"
#include "YABTE/BackTest/Transaction.hpp"
#include "YABTE/BackTest/common.hpp"

using std::map, std::shared_ptr, std::string, std::vector, std::tuple;

using arrow::Table;

namespace YABTE::BackTest {

class Book {
   public:
    Book(const string &name, const string &denom = "USD", double cash = 0.,
         double rate = 0., int interest_round_dp = 3);
    virtual ~Book() = default;
    virtual shared_ptr<Book> clone() const;

    bool test_trades(const vector<shared_ptr<Trade>> &trades) const;
    void add_transactions(const TransactionVector &transactions);
    void eod_tasks(const Timestamp &ts, const DayData &day_data,
                   const AssetMap &asset_map);

    shared_ptr<Table> history() const;

    string name_;
    string denom_;
    double cash_;
    double rate_;
    int interest_round_dp_;
    map<string, double> positions_;
    TransactionVector transactions_;
    vector<tuple<Timestamp, double, double, double>> _history_;
};

using BookMap = map<string, shared_ptr<Book>>;
using BookVector = vector<shared_ptr<Book>>;

}  // namespace YABTE::BackTest