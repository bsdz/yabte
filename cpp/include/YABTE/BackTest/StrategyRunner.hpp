#pragma once

#include <arrow/api.h>
#include <arrow/table.h>
#include <glog/logging.h>
// #include <pybind11/embed.h>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "YABTE/BackTest/Asset.hpp"
#include "YABTE/BackTest/Book.hpp"
#include "YABTE/BackTest/Order.hpp"
#include "YABTE/BackTest/Strategy.hpp"

#ifdef EXPER_PY_SUB_INTERP
#include <pybind11/embed.h>

#include "YABTE/Utilities/Python/Interpreters.hpp"
#endif

using arrow::Table;
using std::shared_ptr, std::string, std::vector, std::variant, std::map,
    std::optional, std::nullopt;

using YABTE::BackTest::ParamMap;

#ifdef EXPER_PY_SUB_INTERP
using YABTE::Utilities::Python::Interpreter,
    YABTE::Utilities::Python::SubInterpreter;
#endif

namespace YABTE::BackTest {

class StrategyRunnerResult {
   public:
    StrategyRunnerResult();

    shared_ptr<Table> book_history() const;

    shared_ptr<OrderDeque> orders_unprocessed_;
    OrderDeque orders_processed_;

    StrategyVector strategies_;
    BookVector books_;
    AssetVector assets_;
};

class StrategyRunner {
   public:
    StrategyRunner(const shared_ptr<Table>& data, const AssetVector& assets,
                   const StrategyVector& strategies, const BookVector& books);

    // #ifdef EXPER_PY_SUB_INTERP
    //     Interpreter interp_;
    // #endif

    StrategyRunnerResult run(const ParamMap& params = {});

    vector<StrategyRunnerResult> run_batch(
        const vector<ParamMap>& params_vector,
        const optional<unsigned int> num_threads = nullopt);

    shared_ptr<Table> data_;
    AssetVector assets_;
    // in python, we accepted types and instantiated them
    // at runtime. to avoid complex template metaprogramming
    // we will instantiate before and attach internal data.
    StrategyVector strategies_;
    BookVector books_;
};

}  // namespace YABTE::BackTest