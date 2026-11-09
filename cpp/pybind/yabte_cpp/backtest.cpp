#include <arrow/python/pyarrow.h>
// #include <pybind11/iostream.h>
#include <pybind11/chrono.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <format>
#include <algorithm> // For std::max

#include "YABTE/Utilities/Arrow/TableHelpers.hpp"
#include <arrow/compute/api.h>

using YABTE::Utilities::Arrow::ComputeMovingAverage;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
#define VERSION_INFO __TIMESTAMP__

#include "./stl_bind_deque.h"
#include "YABTE/BackTest/Asset.hpp"
#include "YABTE/BackTest/Book.hpp"
#include "YABTE/BackTest/Order.hpp"
#include "YABTE/BackTest/PositionalOrder.hpp"
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

                    // If python override returns None, we return nullptr
                    if (o.is_none()) {
                        return nullptr;
                    }

                    auto status = arrow::py::unwrap_table(o.ptr());

                    if (!status.ok()) {
                        throw std::runtime_error(
                            "Error converting pyarrow table to arrow table "
                            "(extend_data): " +
                            status.status().ToString());
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

    void apply(const Timestamp &ts, const DayData &day_data,
               const AssetMap &asset_map) override {
        PYBIND11_OVERRIDE_PURE(void, Order, apply, ts, day_data, asset_map);
    };
};

class SMAXOStrategy : public Strategy {
   public:
    std::string asset_name_;
    int n_;
    int m_;
    
    // Cached column names
    std::string close_col_name_;
    std::string short_ma_col_name_;
    std::string long_ma_col_name_;
    
    // Cached column indices
    int short_ma_col_idx_ = -1;
    int long_ma_col_idx_ = -1;

    SMAXOStrategy(std::string asset_name, int n, int m)
        : Strategy(), asset_name_(asset_name), n_(n), m_(m) {
            this->params_["n"] = n;
            this->params_["m"] = m;
            
            // Pre-calculate column names
            close_col_name_ = std::format("{}.Close", asset_name_);
            short_ma_col_name_ = std::format("{}.CloseSMAShort", asset_name_);
            long_ma_col_name_ = std::format("{}.CloseSMALong", asset_name_);
        }

    shared_ptr<Strategy> clone() const override {
        // Default copy constructor is fine for strings/ints
        return std::make_shared<SMAXOStrategy>(*this);
    }

    shared_ptr<const Table> extend_data(
        const shared_ptr<const Table> &data) override {
        
        auto vals = data->GetColumnByName(close_col_name_);
        if (!vals) {
             // Fallback for tuple names
             auto tuple_name = std::format("('{}', 'Close')", asset_name_);
             vals = data->GetColumnByName(tuple_name);
             if (!vals) {
                 throw std::runtime_error("Column not found: " + close_col_name_ + " or " + tuple_name);
             }
             // Update cached names to tuple format
             short_ma_col_name_ = std::format("('{}', 'CloseSMAShort')", asset_name_);
             long_ma_col_name_ = std::format("('{}', 'CloseSMALong')", asset_name_);
        }

        auto st_cma_short = ComputeMovingAverage(vals, n_);
        auto st_cma_long = ComputeMovingAverage(vals, m_);

        if (!st_cma_short.ok()) throw std::runtime_error("Failed to compute Short MA");
        if (!st_cma_long.ok()) throw std::runtime_error("Failed to compute Long MA");

        shared_ptr<ChunkedArray> ma_chunked_arr_short = st_cma_short.ValueOrDie();
        shared_ptr<ChunkedArray> ma_chunked_arr_long = st_cma_long.ValueOrDie();

        auto f0 = arrow::field(short_ma_col_name_, arrow::float64());
        auto f1 = arrow::field(long_ma_col_name_, arrow::float64());

        auto schema = arrow::schema({f0, f1});
        return arrow::Table::Make(schema, {ma_chunked_arr_short, ma_chunked_arr_long});
    }

    void on_open() override {
        // If indices are not cached, cache them now. 
        // We can trust schema is stable during the run after extend_data.
        if (short_ma_col_idx_ == -1) {
            auto schema = this->data_->schema();
            short_ma_col_idx_ = schema->GetFieldIndex(short_ma_col_name_);
            long_ma_col_idx_ = schema->GetFieldIndex(long_ma_col_name_);
        }
    }

    void on_close() override {
        // Updated logic to use current_idx_ instead of assuming sliced data
        // auto nrows = this->data_->num_rows(); // data_ is now the full table
        auto idx = this->current_idx_;

        if (idx < std::max(n_, m_) + 1) return;

        // Use cached indices if available, else fallback to name lookup (safety)
        std::shared_ptr<arrow::ChunkedArray> s_short, s_long;
        
        if (short_ma_col_idx_ != -1 && long_ma_col_idx_ != -1) {
             s_short = this->data_->column(short_ma_col_idx_);
             s_long = this->data_->column(long_ma_col_idx_);
        } else {
             // Should have been cached in on_open, but just in case
             s_short = this->data_->GetColumnByName(short_ma_col_name_);
             s_long = this->data_->GetColumnByName(long_ma_col_name_);
        }

        if (!s_short || !s_long) return;

        // Helper lambda to get value at index efficiently
        auto get_val = [](const std::shared_ptr<arrow::ChunkedArray>& arr, int64_t i) -> std::optional<double> {
            if (arr->num_chunks() == 1) {
                 // Fast path
                 auto carray = std::static_pointer_cast<arrow::DoubleArray>(arr->chunk(0));
                 if (carray->IsNull(i)) return std::nullopt;
                 return carray->Value(i);
            } else {
                // Chunked path (slower but correct)
                // Use a simplified logic or arrow's resolver. 
                // For this benchmark, assuming likely single chunk or infrequent chunks.
                // Revert to slower scalar access if needed, but let's try to handle it decently.
                int64_t offset = 0;
                for (const auto& chunk : arr->chunks()) {
                    if (i < offset + chunk->length()) {
                        auto carray = std::static_pointer_cast<arrow::DoubleArray>(chunk);
                        auto local_idx = i - offset;
                        if (carray->IsNull(local_idx)) return std::nullopt;
                        return carray->Value(local_idx);
                    }
                    offset += chunk->length();
                }
                return std::nullopt;
            }
        };

        auto v_short_curr = get_val(s_short, idx);
        auto v_short_prev = get_val(s_short, idx - 1);
        auto v_long_curr = get_val(s_long, idx);
        auto v_long_prev = get_val(s_long, idx - 1);

        if (!v_short_curr || !v_short_prev || !v_long_curr || !v_long_prev) return;

        if (v_short_prev.value() < v_long_prev.value() &&
            v_short_curr.value() > v_long_curr.value()) {
            this->orders_->push_back(
                std::make_shared<SimpleOrder>(asset_name_, 100));
        } else if (v_long_prev.value() < v_short_prev.value() &&
                   v_long_curr.value() > v_short_curr.value()) {
            this->orders_->push_back(
                std::make_shared<SimpleOrder>(asset_name_, -100));
        }
    }
};


PYBIND11_MAKE_OPAQUE(AssetMap)
PYBIND11_MAKE_OPAQUE(BookMap)
PYBIND11_MAKE_OPAQUE(OrderDeque)
// PYBIND11_MAKE_OPAQUE(TransactionVector)

PYBIND11_MODULE(_yabte_backtest_lib, m) {
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
        m, "Transaction")
        // Exposed as epoch nanoseconds rather than via pybind11's chrono
        // caster, which renders system_clock::time_point as a *naive local*
        // datetime and so shifts timestamps by the host's UTC offset/DST.
        .def_property_readonly("ts_ns",
                               [](const Transaction &t) {
                                   return std::chrono::duration_cast<
                                              std::chrono::nanoseconds>(
                                              t.ts_.time_since_epoch())
                                       .count();
                               })
        .def_readonly("total", &Transaction::total_)
        .def_readonly("desc", &Transaction::desc_);

    py::class_<CashTransaction, Transaction, shared_ptr<CashTransaction>>(
        m, "CashTransaction")
        .def(py::init<const Timestamp &, const double &, const string &>(),
             py::arg("ts"), py::arg("total"), py::arg("desc") = ""s);

    py::class_<Trade, Transaction, shared_ptr<Trade>>(m, "Trade")
        .def(py::init<const Timestamp &, const double &, const double &,
                      const string &, const optional<string> &>(),

             py::arg("ts"), py::arg("quantity"), py::arg("price"),
             py::arg("asset_name"), py::arg("order_label") = ""s)
        .def_readonly("quantity", &Trade::quantity_)
        .def_readonly("price", &Trade::price_)
        .def_readonly("asset_name", &Trade::asset_name_)
        .def_readonly("order_label", &Trade::order_label_)

        .def("__repr__",
             [](const Trade &t) {
                 return std::format(
                     "<yabte_backtest_cpp.Trade ts={}, total={}, desc={}, "
                     "quantity={}, price={}, asset_name={}, >",
                     t.ts_, t.total_, t.desc_, t.quantity_, t.price_,
                     t.asset_name_);
             })
        .def_property_readonly(
            "total",
            [](const Trade &t) -> py::object {
                // Return total as Decimal("...") to match Python's Trade.total
                // type We use pybind11's eval or import to get decimal.Decimal
                py::object Decimal =
                    py::module_::import("decimal").attr("Decimal");
                // Convert double to string first to avoid float precision
                // issues during init Using 2 decimal places for string
                // formatting as per our rounding logic Or maybe just let Python
                // handle the float string? Let's use format with enough
                // precision then stringify
                std::string s = std::format("{:.2f}", t.total_);
                return Decimal(s);
            })

        ;

    py::bind_vector<TransactionVector>(m, "TransactionVector");

    // book
    py::class_<Book, PyBook, shared_ptr<Book>>(m, "Book")
        .def(py::init([](string name, string denom, double cash, double rate,
                         int interest_round_dp, py::object mandates) {
                 if (!mandates.is_none() && py::len(mandates) > 0) {
                     throw std::runtime_error(
                         "Mandates are not supported in accelerated mode.");
                 }
                 return new Book(name, denom, cash, rate, interest_round_dp);
             }),
             py::arg("name"), py::arg("denom") = "USD", py::arg("cash") = 0.,
             py::arg("rate") = 0., py::arg("interest_round_dp") = 3,
             py::arg("mandates") = py::none())
        .def_readonly("name", &Book::name_)
        .def_readonly("denom", &Book::denom_)
        .def_readonly("cash", &Book::cash_)
        .def_readonly("rate", &Book::rate_)
        .def_readonly("interest_round_dp", &Book::interest_round_dp_)
        .def_readonly("transactions", &Book::transactions_)
        .def_property_readonly("history", [](const Book &b) -> py::handle {
            return arrow::py::wrap_table(b.history());
        });

    py::bind_map<BookMap>(m, "BookMap");

    // asset
    py::class_<Asset, PyAsset, shared_ptr<Asset>>(m, "Asset")
        .def("round_quantity", &Asset::round_quantity);

    py::class_<OHLCAsset, Asset, shared_ptr<OHLCAsset>>(m, "OHLCAsset")
        .def(py::init<const string &, const string &, const optional<int> &,
                      const optional<int> &, const optional<string> &>(),
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

    py::enum_<PositionalOrderCheckType>(m, "PositionalOrderCheckType")
        .value("POS_TQ_DIFFER", PositionalOrderCheckType::POS_TQ_DIFFER)
        .value("ZERO_POS", PositionalOrderCheckType::ZERO_POS)
        .export_values();

    py::class_<SimpleOrder, Order, shared_ptr<SimpleOrder>>(m, "SimpleOrder")
        .def(py::init<const string &, const double &, const OrderSizeType &,
                      const optional<string> &, const optional<string> &,
                      const int, const optional<string> &>(),
             py::arg("asset_name"), py::arg("size"),
             py::arg("size_type") = OrderSizeType::QUANTITY,
             py::arg("book_name") = nullopt, py::arg("label") = nullopt,
             py::arg("priority") = 0, py::arg("key") = nullopt

        );

    py::class_<PositionalOrder, SimpleOrder, shared_ptr<PositionalOrder>>(
        m, "PositionalOrder")
        .def(py::init<const string &, const double &, const OrderSizeType &,
                      const PositionalOrderCheckType &,
                      const optional<string> &, const optional<string> &,
                      const int, const optional<string> &>(),
             py::arg("asset_name"), py::arg("size"),
             py::arg("size_type") = OrderSizeType::QUANTITY,
             py::arg("check_type") = PositionalOrderCheckType::POS_TQ_DIFFER,
             py::arg("book_name") = nullopt, py::arg("label") = nullopt,
             py::arg("priority") = 0, py::arg("key") = nullopt);

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
                                   return arrow::py::wrap_table(
                                       std::const_pointer_cast<arrow::Table>(
                                           s.data_));
                               })
        .def_property_readonly(
            "asset_map", [](const Strategy &s) { return s.asset_map_.get(); })
        .def_property_readonly(
            "orders", [](const Strategy &s) { return s.orders_.get(); })
        .def_readonly("params", &Strategy::params_)
        .def_readonly("current_idx", &Strategy::current_idx_);

    py::class_<SMAXOStrategy, Strategy, shared_ptr<SMAXOStrategy>>(m, "SMAXOStrategy")
        .def(py::init<std::string, int, int>(),
             py::arg("asset_name"), py::arg("n"), py::arg("m"));

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
                    "Error converting pyarrow table to arrow table "
                    "(constructor): " +
                    status.status().ToString());
            }
            std::shared_ptr<arrow::Table> data = status.ValueOrDie();

            return new StrategyRunner(data, assets, strategies, books);
        }))
        // Release the GIL so other Python threads (e.g. a user's thread pool)
        // can run alongside the C++ event loop. Python strategy callbacks
        // re-acquire it through the PyStrategy trampolines.
        .def("run", &StrategyRunner::run, py::arg("params") = ParamMap{},
             py::call_guard<py::gil_scoped_release>())
        .def("run_batch", &StrategyRunner::run_batch,
             py::call_guard<py::gil_scoped_release>())
        // .def("run_batch",
        //      [](StrategyRunner &sr, pybind11::iterable py_iterable) {
        //          auto temp = py_iterable.cast<ParamMap>();
        //          return sr.run_batch(temp);
        //      })
        ;
}
