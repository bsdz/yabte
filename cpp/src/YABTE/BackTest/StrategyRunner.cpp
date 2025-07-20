#include "YABTE/BackTest/StrategyRunner.hpp"

#include <BS_thread_pool.hpp>
#include <ranges>
#include <unordered_map>
#include <utility>

#include "YABTE/Utilities/Arrow/TableHelpers.hpp"
#ifdef EXPER_PY_SUB_INTERP
#include "YABTE/Utilities/Python/Interpreters.hpp"
#endif

using std::dynamic_pointer_cast, std::make_shared;

using YABTE::Utilities::Arrow::ExtendTable,
    YABTE::Utilities::Arrow::HorizConcatTables;

#ifdef EXPER_PY_SUB_INTERP
using YABTE::Utilities::Python::ThreadsAllowedScope;
#endif

namespace YABTE::BackTest {

#ifdef EXPER_PY_SUB_INTERP
thread_local SubInterpreter subinterp_;
#endif

StrategyRunnerResult::StrategyRunnerResult()
    : orders_unprocessed_(make_shared<OrderDeque>()) {}

shared_ptr<Table> StrategyRunnerResult::book_history() const {
    vector<string> book_names;
    vector<shared_ptr<const Table>> book_history_tables;
    for (auto& book : this->books_) {
        book_names.push_back(book->name_);
        book_history_tables.push_back(book->history());
    }
    auto st_et = HorizConcatTables(book_history_tables, book_names);
    CHECK(st_et.ok()) << "Error: " << st_et.status();
    return st_et.ValueOrDie();
}

StrategyRunner::StrategyRunner(const shared_ptr<Table>& data,
                               const AssetVector& assets,
                               const StrategyVector& strategies,
                               const BookVector& books)
    : data_(data), assets_(assets), strategies_(strategies), books_(books) {}

StrategyRunnerResult StrategyRunner::run(const ParamMap& params) {
#ifdef EXPER_PY_SUB_INTERP
    auto interp = subinterp_.interp();
    SubInterpreter::ThreadScope scope(interp);
#endif

    DLOG(INFO) << "Running strategy runner";
    StrategyRunnerResult result;

    // copy books and stategies (assets are immutable but copy anyway)
    for (auto& s : this->strategies_) {
        result.strategies_.push_back(shared_ptr<Strategy>(s->clone()));
    }

    for (auto& b : this->books_)
        result.books_.push_back(shared_ptr<Book>(b->clone()));

    for (auto& a : this->assets_)
        result.assets_.push_back(shared_ptr<Asset>(a->clone()));

    // generate asset and book maps
    shared_ptr<AssetMap> asset_map = make_shared<AssetMap>();
    shared_ptr<BookMap> book_map = make_shared<BookMap>();
    for (auto a : result.assets_) asset_map->emplace(a->name_, a);
    for (auto b : result.books_) book_map->emplace(b->name_, b);

    auto default_book = result.books_[0];

    auto calendar = this->data_->GetColumnByName("Date");
    int64_t nr = this->data_->num_rows();

    std::unordered_map<shared_ptr<Strategy>, shared_ptr<const Table>> data_map;

    // init
    for (auto& strategy : result.strategies_) {
        strategy->asset_map_ = asset_map;
        strategy->book_map_ = book_map;
        strategy->orders_ = result.orders_unprocessed_;
        strategy->params_ = params;

        // merge tables here (using pointers to avoid copying data)
        auto new_data = strategy->extend_data(this->data_);
        if (new_data) {
            DLOG(INFO) << "Extending data for strategy";
            auto st_et = ExtendTable(this->data_, new_data);
            CHECK(st_et.ok()) << "Error: " << st_et.status();
            data_map.insert(std::make_pair(
                strategy, dynamic_pointer_cast<Table>(st_et.ValueOrDie())));
        } else {
            DLOG(INFO) << "Not extending data for strategy";
            data_map.insert(std::make_pair(strategy, this->data_));
        }

        // run strategy's init
        strategy->init();
    }

    // run event loop
    for (const auto [i, ts] :
         arrow::stl::Iterate<arrow::TimestampType>(*calendar) |
             std::ranges::views::enumerate) {
        if (!ts.has_value()) continue;

        auto ts_chrono = timestamp_from_ns(*ts);
        auto day_data = this->data_->Slice(i, 1);

        vector<shared_ptr<Order>> orders_next_ts;

        // open
        for (auto& strategy : result.strategies_) {
            // TODO: mask out non-open available data
            strategy->data_ = data_map[strategy]->Slice(0, i + 1);
            strategy->on_open();
        }

        // process orders
        while (!result.orders_unprocessed_->empty()) {
            auto order = result.orders_unprocessed_->front();
            result.orders_unprocessed_->pop_front();

            // set book attribute if needed
            if (!order->book_) {
                if (order->book_name_.has_value()) {
                    order->book_ = book_map->at(order->book_name_.value());
                } else {
                    // fall back to first available book
                    order->book_ = default_book;
                }
            }

            order->apply(ts_chrono, *day_data, *asset_map);

            // add any child orders to next ts
            orders_next_ts.insert(orders_next_ts.end(),
                                  order->suborders_.begin(),
                                  order->suborders_.end());

            result.orders_processed_.push_back(order);
        }

        // extend with orders for next ts
        result.orders_unprocessed_->insert(result.orders_unprocessed_->end(),
                                           orders_next_ts.begin(),
                                           orders_next_ts.end());

        // close
        for (auto& strategy : result.strategies_) {
            strategy->on_close();
        }

        // run book end-of-day tasks
        for (auto& book : result.books_) {
            book->eod_tasks(ts_chrono, *day_data, *asset_map);
        }
    }

    DLOG(INFO) << "Finished running strategy runner";
    return result;
}

vector<StrategyRunnerResult> StrategyRunner::run_batch(

    const vector<ParamMap>& params_vector,
    const optional<unsigned int> num_threads) {
    unsigned int tp_num_threads = 1;
    if (num_threads.has_value()) {
        tp_num_threads = num_threads.value();
    } else {
        tp_num_threads = params_vector.size();
    }

#ifdef EXPER_PY_SUB_INTERP
    Interpreter interp;
    // vector<SubInterpreter> subinterps(tp_num_threads);
    pybind11::gil_scoped_acquire gil;
    SubInterpreter foo;
    ThreadsAllowedScope t1;
#endif

    BS::thread_pool pool{tp_num_threads};

    // #ifdef EXPER_PY_SUB_INTERP
    //     Interpreter interp;
    //     ThreadsAllowedScope t1;
    //     vector<SubInterpreter> subinterps(tp_num_threads);
    // #endif

    BS::multi_future<StrategyRunnerResult> sequence_future =
        pool.submit_sequence<int>(0, params_vector.size(),
                                  [this, &params_vector](int i) {
                                      return this->run(params_vector[i]);
                                  });
    std::vector<StrategyRunnerResult> results = sequence_future.get();
    return results;
}

}  // namespace YABTE::BackTest