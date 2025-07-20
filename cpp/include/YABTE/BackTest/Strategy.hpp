#pragma once

#include <arrow/table.h>
#include <pybind11/embed.h>

#include <map>
#include <memory>
#include <string>

#include "YABTE/BackTest/Asset.hpp"
#include "YABTE/BackTest/Book.hpp"
#include "YABTE/BackTest/Order.hpp"

using arrow::Table;
using std::make_shared, std::shared_ptr, std::string, std::variant, std::map;

namespace YABTE::BackTest {

using ParamMap = map<string, variant<double, string, bool, int>>;

// allow variants to be printed
template <class Var, class = std::variant_alternative_t<0, Var>>
std::ostream& operator<<(std::ostream& os, Var const& v);

// allow param maps to be printed
std::ostream& operator<<(std::ostream& os, ParamMap const& v);

class Strategy {
   public:
    virtual ~Strategy() = default;
    virtual shared_ptr<Strategy> clone() const = 0;
    bool operator==(Strategy const& rhs) const;

    virtual void init();
    virtual void on_open();
    virtual void on_close();

    // runner keeps a map or each strategies extended data
    virtual shared_ptr<const Table> extend_data(
        const shared_ptr<const Table>& data);

    // attached/copied during strategy runner
    ParamMap params_;
    shared_ptr<AssetMap> asset_map_;
    shared_ptr<BookMap> book_map_;
    shared_ptr<OrderDeque> orders_;

    // writable during init, readable during open/close
    shared_ptr<Table> data_ = nullptr;

    //    protected:
    Strategy() = default;
    // Strategy(const Strategy&){};
};

using StrategyVector = std::vector<shared_ptr<Strategy>>;

// CRTP pattern for cloning strategies
// template <class Derived>
// class CloneableStrategy : public Strategy {
//    public:
//     virtual shared_ptr<Strategy> clone() const {
//         return make_shared<const Derived>(*this);  // call the copy ctor.
//     }
// };

}  // namespace YABTE::BackTest
