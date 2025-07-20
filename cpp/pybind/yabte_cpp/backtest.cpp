#include <arrow/python/pyarrow.h>
// #include <pybind11/iostream.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <format>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
#define VERSION_INFO __TIMESTAMP__

#include "./stl_bind_deque.h"
#include "YABTE/BackTest/Asset.hpp"
#include "YABTE/BackTest/Book.hpp"
#include "YABTE/BackTest/Order.hpp"
#include "YABTE/BackTest/Strategy.hpp"
#include "YABTE/BackTest/StrategyRunner.hpp"
#include "YABTE/BackTest/Transaction.hpp"
#include "YABTE/Utilities/Arrow/TableHelpers.hpp"

using namespace YABTE::BackTest;

using std::vector, std::tuple, std::shared_ptr, std::optional, std::nullopt,
    std::make_shared, std::string;

namespace py = pybind11;

class PyTransaction : public Transaction {
   public:
    using Transaction::Transaction;

    shared_ptr<Transaction> clone() const override {
        auto self = py::cast(this);
        auto cloned = self.attr("clone")();

        auto keep_python_state_alive = make_shared<py::object>(cloned);
        auto ptr = cloned.cast<PyTransaction *>();

        return shared_ptr<Transaction>(keep_python_state_alive, ptr);
    }
};

// For cloning idioms, see:
// https://github.com/pybind/pybind11/issues/1049#issuecomment-326688270

class PyStrategy : public Strategy {
   public:
    using Strategy::Strategy;
    // PyStrategy(const Strategy &s) : Strategy(s) {}
    PyStrategy(const Strategy &s) {}

    shared_ptr<Strategy> clone() const override {
        py::gil_scoped_acquire gil;
        auto self = py::cast(this);
        auto cloned = self.attr("clone")();

        auto keep_python_state_alive = make_shared<py::object>(cloned);
        auto ptr = cloned.cast<PyStrategy *>();

        return shared_ptr<Strategy>(keep_python_state_alive, ptr);
    }

    void init() override { PYBIND11_OVERRIDE(void, Strategy, init); };
    void on_open() override { PYBIND11_OVERRIDE(void, Strategy, on_open); };
    void on_close() override { PYBIND11_OVERRIDE(void, Strategy, on_close); };

    shared_ptr<const Table> extend_data(
        // inline PYBIND11_OVERRIDE macro and adjust wrap/unwrap
        // arrow table input / output.
        const shared_ptr<const Table> &data) override {
        pybind11::gil_scoped_acquire gil;
        std::shared_ptr<Table> data_nc = std::const_pointer_cast<Table>(
            const_cast<std::shared_ptr<const Table> &>(data));

        py::handle data_nc_w = arrow::py::wrap_table(data_nc);

        do {
            do {
                pybind11::gil_scoped_acquire gil;
                pybind11::function override = pybind11::get_override(
                    static_cast<const Strategy *>(this), "extend_data");
                if (override) {
                    auto o = override(data_nc_w);

                    auto status = arrow::py::unwrap_table(o.ptr());

                    if (!status.ok()) {
                        throw std::runtime_error(
                            "Error converting pyarrow table to arrow table");
                    }
                    std::shared_ptr<const arrow::Table> o_uw =
                        status.ValueOrDie();

                    return std::move(o_uw);
                }
            } while (false);
            return Strategy::extend_data(data);
        } while (false);
    };
};

class PyAsset : public Asset {
   public:
    using Asset::Asset;

    shared_ptr<Asset> clone() const override {
        auto self = py::cast(this);
        auto cloned = self.attr("clone")();

        auto keep_python_state_alive = make_shared<py::object>(cloned);
        auto ptr = cloned.cast<PyAsset *>();

        return shared_ptr<Asset>(keep_python_state_alive, ptr);
    }

    double intraday_traded_price(
        const DayData &asset_day_data,
        const optional<double> size = nullopt) const override {
        PYBIND11_OVERRIDE_PURE(double, Asset, intraday_traded_price,
                               asset_day_data, size);
    }

    double end_of_day_price(const DayData &asset_day_data) const override {
        PYBIND11_OVERRIDE_PURE(double, Asset, end_of_day_price, asset_day_data);
    };

    vector<tuple<string, AssetDataFieldInfo>> data_fields() const override {
        PYBIND11_OVERRIDE_PURE_NAME(
            PYBIND11_TYPE(vector<tuple<string, AssetDataFieldInfo>>),
            PYBIND11_TYPE(Asset), "data_fields", data_fields);
    };
};

class PyBook : public Book {
   public:
    using Book::Book;

    shared_ptr<Book> clone() const override {
        auto self = py::cast(this);
        auto cloned = self.attr("clone")();

        auto keep_python_state_alive = make_shared<py::object>(cloned);
        auto ptr = cloned.cast<PyBook *>();

        return shared_ptr<Book>(keep_python_state_alive, ptr);
    }
};

class PyOrder : public Order {
   public:
    using Order::Order;
    PyOrder(const Order &s) {}

    shared_ptr<Order> clone() const override {
        auto self = py::cast(this);
        auto cloned = self.attr("clone")();

        auto keep_python_state_alive = make_shared<py::object>(cloned);
        auto ptr = cloned.cast<PyOrder *>();

        return shared_ptr<Order>(keep_python_state_alive, ptr);
    }

    void post_complete(vector<shared_ptr<Trade>> trades) override {
        PYBIND11_OVERRIDE(void, Order, post_complete, trades);
    };

    void apply(Timestamp &ts, DayData &day_data, AssetMap &asset_map) {
        PYBIND11_OVERRIDE_PURE(void, Order, apply, ts, day_data, asset_map);
    };
};

PYBIND11_MAKE_OPAQUE(AssetMap)
PYBIND11_MAKE_OPAQUE(BookMap)
PYBIND11_MAKE_OPAQUE(OrderDeque)
// PYBIND11_MAKE_OPAQUE(TransactionVector)

PYBIND11_MODULE(yabte_cpp_backtest, m) {
    arrow::py::import_pyarrow();

    m.doc() = R"pbdoc(
        YABTE C++ Backtest
        -----------------------

        .. currentmodule:: yabte_cpp_backtest

        .. autosummary::
           :toctree: _generate

    )pbdoc";

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    // py::add_ostream_redirect(m, "ostream_redirect");

    // transaction
    py::class_<Transaction, PyTransaction, shared_ptr<Transaction>>(
        m, "Transaction");

    py::class_<CashTransaction, Transaction, shared_ptr<CashTransaction>>(
        m, "CashTransaction")
        .def(py::init<const Timestamp &, const double &, const string &>(),
             py::arg("ts"), py::arg("total"), py::arg("desc") = ""s);

    py::class_<Trade, Transaction, shared_ptr<Trade>>(m, "Trade")
        .def(py::init<const Timestamp &, const double &, const double &,
                      const string &, const optional<string> &>(),

             py::arg("ts"), py::arg("quantity"), py::arg("price"),
             py::arg("asset_name"), py::arg("order_label") = ""s)

        .def("__repr__",
             [](const Trade &t) {
                 return std::format(
                     "<yabte_backtest_cpp.Trade ts={}, total={}, desc={}, "
                     "quantity={}, price={}, asset_name={}, >",
                     t.ts_, t.total_, t.desc_, t.quantity_, t.price_,
                     t.asset_name_);
             })

        ;

    py::bind_vector<TransactionVector>(m, "TransactionVector");

    // book
    py::class_<Book, PyBook, shared_ptr<Book>>(m, "Book")
        .def(py::init<string, string, double, double, int>(), py::arg("name"),
             py::arg("denom") = "USD", py::arg("cash") = 0.,
             py::arg("rate") = 0., py::arg("interest_round_dp") = 3)
        .def_readonly("transactions", &Book::transactions_)
        .def_property_readonly("history", [](const Book &b) -> py::handle {
            return arrow::py::wrap_table(b.history());
        });

    py::bind_map<BookMap>(m, "BookMap");

    // asset
    py::class_<Asset, PyAsset, shared_ptr<Asset>>(m, "Asset");

    py::class_<OHLCAsset, Asset, shared_ptr<OHLCAsset>>(m, "OHLCAsset")
        .def(py::init<const string &, const string &, const int, const int,
                      const optional<string> &>(),
             py::arg("name"), py::arg("denom") = "USD",
             py::arg("price_round_dp") = 2, py::arg("quantity_round_dp") = 2,
             py::arg("data_label") = nullopt)
        .def(py::init<const string &, const string &>());

    py::enum_<AssetDataFieldInfo>(m, "AssetDataFieldInfo")
        .value("AVAILABLE_AT_CLOSE", AssetDataFieldInfo::AVAILABLE_AT_CLOSE)
        .value("AVAILABLE_AT_OPEN", AssetDataFieldInfo::AVAILABLE_AT_OPEN)
        .value("REQUIRED", AssetDataFieldInfo::REQUIRED)
        .export_values();

    py::bind_map<AssetMap>(m, "AssetMap");

    // order
    py::enum_<OrderSizeType>(m, "OrderSizeType")
        .value("QUANTITY", OrderSizeType::QUANTITY)
        .value("NOTIONAL", OrderSizeType::NOTIONAL)
        .value("BOOK_PERCENT", OrderSizeType::BOOK_PERCENT)
        .export_values();

    py::class_<Order, PyOrder, shared_ptr<Order>>(m, "Order");

    py::class_<SimpleOrder, Order, shared_ptr<SimpleOrder>>(m, "SimpleOrder")
        .def(py::init<const string &, const double &, const OrderSizeType &,
                      const optional<string> &, const optional<string> &,
                      const int, const optional<string> &>(),
             py::arg("asset_name"), py::arg("size"),
             py::arg("size_type") = OrderSizeType::QUANTITY,
             py::arg("book_name") = nullopt, py::arg("label") = nullopt,
             py::arg("priority") = 0, py::arg("key") = nullopt

        );

    py::bind_deque<OrderDeque>(m, "OrderDeque");

    // strategy
    py::class_<Strategy, PyStrategy, shared_ptr<Strategy>>(m, "Strategy")
        .def(py::init<>())
        .def(py::init<const Strategy &>())
        .def("extend_data",
             [](Strategy &s, pybind11::object py_table) {
                 auto status = arrow::py::unwrap_table(py_table.ptr());

                 if (!status.ok()) {
                     throw std::runtime_error(
                         "Error converting pyarrow table to arrow table");
                 }
                 std::shared_ptr<arrow::Table> data_uw = status.ValueOrDie();
                 return s.extend_data(data_uw);
             })
        .def_property_readonly("data",
                               [](const Strategy &s) -> py::handle {
                                   return arrow::py::wrap_table(s.data_);
                               })
        .def_property_readonly(
            "asset_map", [](const Strategy &s) { return s.asset_map_.get(); })
        .def_property_readonly(
            "orders", [](const Strategy &s) { return s.orders_.get(); })
        .def_readonly("params", &Strategy::params_);

    // strategy runner
    py::bind_map<ParamMap>(m, "ParamMap");

    py::class_<StrategyRunnerResult>(m, "StrategyRunnerResult")
        .def(py::init<>())
        .def_property_readonly("orders_unprocessed",
                               [](const StrategyRunnerResult &srr) {
                                   return srr.orders_unprocessed_.get();
                               })
        .def_readonly("orders_processed",
                      &StrategyRunnerResult::orders_processed_)
        .def_readonly("books", &StrategyRunnerResult::books_)
        .def_property_readonly(
            "book_history", [](const StrategyRunnerResult &srr) -> py::handle {
                return arrow::py::wrap_table(srr.book_history());
            });

    py::class_<StrategyRunner>(m, "StrategyRunner")
        .def(py::init([](pybind11::object py_table, const AssetVector &assets,
                         const StrategyVector &strategies,
                         const BookVector &books) {
            auto status = arrow::py::unwrap_table(py_table.ptr());

            if (!status.ok()) {
                throw std::runtime_error(
                    "Error converting pyarrow table to arrow table");
            }
            std::shared_ptr<arrow::Table> data = status.ValueOrDie();

            return new StrategyRunner(data, assets, strategies, books);
        }))
        .def("run", &StrategyRunner::run)
        .def("run_batch", &StrategyRunner::run_batch,
             py::call_guard<py::gil_scoped_release>())
        // .def("run_batch",
        //      [](StrategyRunner &sr, pybind11::iterable py_iterable) {
        //          auto temp = py_iterable.cast<ParamMap>();
        //          return sr.run_batch(temp);
        //      })
        ;
}