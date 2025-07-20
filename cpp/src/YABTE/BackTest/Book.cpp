#include "YABTE/BackTest/Book.hpp"

#include <arrow/stl.h>

#include <cmath>

#include "YABTE/Utilities/Arrow/TableHelpers.hpp"

using std::make_shared;

namespace YABTE::BackTest {

Book::Book(const string& name, const string& denom, double cash, double rate,
           int interest_round_dp)
    : name_(name),
      denom_(denom),
      cash_(cash),
      rate_(rate),
      interest_round_dp_(interest_round_dp) {}

shared_ptr<Book> Book::clone() const { return make_shared<Book>(*this); }

bool Book::test_trades(const vector<shared_ptr<Trade>>& trades) const {
    // TODO: Implement this function
    // for asset_name, asset_trades in groupby(trades, lambda t: t.asset_name):
    //     if asset_name in self.mandates:
    //         total_quantity = sum(t.quantity for t in asset_trades)
    //         if not self.mandates[asset_name].check(
    //             self.positions[asset_name], total_quantity
    //         ):
    //             return False

    return true;
}

void Book::add_transactions(const TransactionVector& transactions) {
    for (auto& tran : transactions) {
        if (auto trade = dynamic_cast<const Trade*>(tran.get());
            trade != nullptr) {
            this->positions_[trade->asset_name_] += trade->quantity_;
            this->cash_ += trade->total_;
        } else if (auto ctran =
                       dynamic_cast<const CashTransaction*>(tran.get());
                   ctran != nullptr) {
            this->cash_ += ctran->total_;
        } else {
            throw std::runtime_error("Unsupport transaction class");
        }

        this->transactions_.push_back(tran);
    }
}

void Book::eod_tasks(const Timestamp& ts, const DayData& day_data,
                     const AssetMap& asset_map) {
    // Run end of day tasks such as book keeping."""
    // accumulate continously compounded interest

    auto interest = round_n_digits(this->cash_ * (std::exp(this->rate_) - 1),
                                   this->interest_round_dp_);
    if (this->rate_ != 0 && interest != 0) {
        auto cash_trans = CashTransaction(
            ts, interest, "interest payment on cash {self.cash:.2f}"s);
        const TransactionVector trans = {make_shared<Transaction>(cash_trans)};
        this->add_transactions(trans);
    }

    auto mtm = 0.0;
    for (const auto& [an, q] : this->positions_) {
        if (auto asset = asset_map.at(an); asset != nullptr) {
            mtm += asset->end_of_day_price(*asset->_filter_data(day_data)) * q;
        }
    }

    this->_history_.push_back({ts, this->cash_, mtm, this->cash_ + mtm});
    // cash = float(self.cash)
    // mtm = float(
    //     sum(
    //         asset.end_of_day_price(asset._filter_data(day_data)) * q
    //         for an, q in self.positions.items()
    //         if (asset := asset_map.get(an))
    //     )
    // )
    // self._history.append([ts, cash, mtm, cash + mtm])
}

shared_ptr<Table> Book::history() const {
    shared_ptr<Table> table;
    vector<std::string> names = {"ts", "cash", "mtm", "total"};
    if (!arrow::stl::TableFromTupleRange(arrow::default_memory_pool(),
                                         this->_history_, names, &table)
             .ok()) {
        throw std::runtime_error("Error creating table");
    }
    return table;
}

}  // namespace YABTE::BackTest