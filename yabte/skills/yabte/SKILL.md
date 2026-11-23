---
name: yabte
description: Use when writing, running or debugging trading strategy backtests with the yabte Python library (StrategyRunner, Strategy, SimpleOrder, PositionalOrder, Book, OHLCAsset), choosing between its Python and C++ engines, sweeping parameters with run_batch, or using its portfolio optimisation and price simulation utilities.
---

# Backtesting with yabte

yabte is an event-driven backtester. You give a `StrategyRunner` price data, assets,
strategies and books. For each row of the data it calls each strategy's `on_open`,
executes queued orders, then calls `on_close`. It returns a `StrategyRunnerResult` with
the trades and the daily book values.

Everything in this file runs as written. The `python` blocks run in order and share
their names, and a test in the yabte repo executes them.

## Data and assets

`data` is a pandas DataFrame with:

- a `DatetimeIndex` that is unique and increasing. A row can be a day or an intraday
  bar.
- two column levels, `(asset, field)`, with columns for every asset passed to the
  runner. Other columns are dropped.

`OHLCAsset` reads `Open`, `High`, `Low`, `Close` and `Volume`; only `Close` is
required. Its `data_label` (default: its `name`) picks its columns.

```python
import numpy as np
import pandas as pd

from yabte.backtest import (
    Book,
    CostType,
    OHLCAsset,
    OrderSizeType,
    PositionalOrder,
    SimpleOrder,
    Strategy,
    StrategyRunner,
    find_missing_data,
)

rng = np.random.default_rng(1)
index = pd.bdate_range("2024-01-01", periods=250)
columns = {}
for name in ["ACME", "BOLT"]:
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(index))))
    columns[(name, "Open")] = close * (1 + rng.normal(0, 0.002, len(index)))
    columns[(name, "High")] = close * 1.01
    columns[(name, "Low")] = close * 0.99
    columns[(name, "Close")] = close
data = pd.DataFrame(columns, index=index)
assets = [OHLCAsset(name="ACME"), OHLCAsset(name="BOLT")]
```

A missing price raises `MissingPriceError` (a `ValueError`) as soon as a run needs it:
an order executing on that row, or a held position valued at the close. Gaps nothing
uses are fine. List gaps in required fields before a run:

```python
assert find_missing_data(data, assets).empty  # columns: ts, asset_name, field
```

## Writing a strategy

Subclass `Strategy` and override any of:

- `init()`: runs once before the first row. `self.data` is the whole dataset, and you
  may assign to it to add columns such as indicators.
- `on_open()`: runs at each row's open. Same-row fields not available at the open
  (`High`, `Low`, `Close`, `Volume` for `OHLCAsset`) are NaN, so you can't look ahead.
  Orders placed here execute on this row.
- `on_close()`: runs at each row's close, with the whole row visible. Orders placed here
  execute on the next row.

Inside these, use:

- `self.data`: the data up to and including `self.ts`, the current timestamp;
- `self.params`: a `pd.Series` of the run's parameters;
- `self.orders`: the queue to append orders to;
- `self.books` and `self.assets`: dicts by name;
- `self.record(**values)`: stores diagnostics against `self.ts`. Read them after the
  run as a DataFrame, `srr.strategies[0].recorded`, one row per timestamp.

```python
class SMACross(Strategy):
    def init(self):
        p = self.params
        closes = self.data.loc[:, (slice(None), "Close")]
        fast = closes.rolling(p.get("fast", 10)).mean()
        slow = closes.rolling(p.get("slow", 30)).mean()
        self.data = pd.concat(
            [
                self.data,
                fast.rename({"Close": "Fast"}, axis=1, level=1),
                slow.rename({"Close": "Slow"}, axis=1, level=1),
            ],
            axis=1,
        ).sort_index(axis=1)

    def on_close(self):
        for name in self.assets:
            row = self.data[name].iloc[-1]
            if np.isnan(row.Slow):
                continue  # still warming up
            self.record(**{f"{name}_spread": row.Fast - row.Slow})
            target = 100 if row.Fast > row.Slow else 0
            # trades to the target position; repeating a target does nothing
            self.orders.append(PositionalOrder(asset_name=name, size=target, key=name))
```

## Running and reading results

```python
runner = StrategyRunner(
    data=data,
    assets=assets,
    strategies=[SMACross()],
    books=[Book(name="Main", cash="100000")],
)
srr = runner.run({"fast": 10, "slow": 30})

srr.book_history.Main.total  # one row per date; columns (book, field): cash, mtm, total
srr.transaction_history  # ts, total, desc, quantity, price, asset_name, ..., book, type
statuses = [o.status.name for o in srr.orders_processed]
spreads = srr.strategies[0].recorded  # what SMACross passed to self.record
```

`run()` copies the books, assets and strategies, so the same runner can run again with
other parameters. `transaction_history` holds trades, transaction costs (`Cost`) and
interest payments (`CashTransaction`, which have no `asset_name`), and its `type` column
says which. Only trades have a `quantity`, so filter on `type == "Trade"` or sum
`quantity` for positions; costs have a `charged_quantity` instead.

## Orders

All orders take `book` (a book name; omit it for the first book), `label`, `priority`
(higher executes first within a row) and `key`. A newer queued order with the same
`key` replaces an older one, which ends with status `REPLACED`.

- `SimpleOrder(asset_name, size, size_type=...)` trades `size`, sized by
  `OrderSizeType`:
  - `QUANTITY` (the default): units.
  - `NOTIONAL`: money.
  - `BOOK_PERCENT`: a percentage of the book's **cash**, not its total value.
- `PositionalOrder(...)` takes the same arguments, but `size` is the target position.
  `check_type` decides whether it trades: `PositionalOrderCheckType.POS_TQ_DIFFER` (the
  default) when the position differs from the target, or `ZERO_POS` only when the
  position is zero. By default it closes the current position and opens the new one,
  booking two trades. With `close_first=False` it books one trade for the difference,
  which suits daily rebalancing such as a delta hedge.
- `BasketOrder(asset_names, weights, size, size_type)` and `PositionalBasketOrder`
  trade several assets at once. With `BOOK_PERCENT`, `weights` are percentages of cash
  plus positions at the traded prices, and `size` is ignored. They are Python engine
  only.

Orders fill at the asset's `intraday_traded_price`. For `OHLCAsset` that depends on its
`fill_price`:

- `"midpoint"` (the default): the mean of `High` and `Low`, or `Close` if either is
  missing. Neither is known at the open, so orders placed in `on_open` fill at prices
  from later in the row.
- `"open"`: `Open`, which is then a required field. Use it for orders placed in
  `on_open`.
- `"close"`: `Close`.

Fills carry no costs unless the asset sets them: `cost` plus a `CostType` in
`cost_type`, for example `OHLCAsset(name="ACME", cost="5", cost_type=CostType.BPS)`.

- `BPS` (the default type): basis points of the traded notional.
- `PER_UNIT`: per unit of quantity, so per contract if you trade contracts.
- `PER_ORDER`: a flat fee.

Each order is charged on its net quantity in each asset, so a `PositionalOrder` that
closes and reopens pays only on the difference. The cost is booked as a `Cost`
transaction beside the trades and comes out of cash, so `book_history` and later
`BOOK_PERCENT` orders see it. Costs aren't rounded. For anything else, such as tiered
fees, override `transaction_cost(quantity, price)` (Python engine only).

Final statuses are `COMPLETE`,
`CANCELLED`, `REPLACED` and `MANDATE_FAILED`. An order still `OPEN` at the end of the
run is in `srr.orders_unprocessed`.

To make limit, stop or bracket orders, subclass an order. These hooks run on the
Python engine only:

- `pre_execute_check(ts, trade_price)`: return `None` to fill, `OrderStatus.OPEN` to
  try again on the next row, or `OrderStatus.CANCELLED`.
- `post_complete(trades)`: append follow-on orders to `self.suborders`.

```python
from dataclasses import dataclass
from decimal import Decimal

from yabte.backtest import OrderStatus


@dataclass(kw_only=True)
class LimitBuy(SimpleOrder):
    limit: Decimal

    def pre_execute_check(self, ts, trade_price):
        return None if trade_price <= self.limit else OrderStatus.OPEN


class BuyTheDip(Strategy):
    def on_close(self):
        if len(self.data) == 1:
            close = Decimal(str(self.data.ACME.Close.iloc[-1]))
            self.orders.append(
                LimitBuy(asset_name="ACME", size=10, limit=close * Decimal("0.95"))
            )


dip = StrategyRunner(
    data=data,
    assets=assets,
    strategies=[BuyTheDip()],
    books=[Book(name="Main", cash="100000")],
).run()
```

## Books

`Book(name, cash, rate=0, interest_round_dp=3, mandates={})`:

- **Cash** may go negative. Several books can share one runner, each with its own cash,
  positions and result columns.
- **`rate`** is a daily, continuously compounded rate. Each date the book receives
  `cash * (exp(rate) - 1)`, so use `annual_rate / 252` for trading-day data.
- **Mandates** reject trades before they're booked (status `MANDATE_FAILED`). A mandate
  is a `BookMandate` subclass whose `check(current_pos, quantity)` returns `False` to
  reject, keyed by asset name. They are experimental and Python engine only.

## Engines

`StrategyRunner(..., engine="python")` is the reference implementation. The default,
`"auto"`, currently means `"python"`. `engine="cpp"` runs the event loop in the compiled
extension and gives the same results for supported features. Python strategies run on
it unchanged. Check it is available with
`from yabte.backtest.runner_cpp import yabte_cpp` (`None` when it isn't built).

```python
from yabte.backtest.runner_cpp import yabte_cpp

if yabte_cpp is not None:
    srr_cpp = StrategyRunner(
        data=data,
        assets=assets,
        strategies=[SMACross()],
        books=[Book(name="Main", cash="100000")],
        engine="cpp",
    ).run({"fast": 10, "slow": 30})
    assert np.allclose(
        srr_cpp.book_history.Main.total.astype(float),
        srr.book_history.Main.total.astype(float),
    )
```

The C++ engine's limits:

- Mandates raise `ValueError`.
- Basket orders raise `NotImplementedError`.
- Asset subclasses that override `intraday_traded_price`, `end_of_day_price`,
  `data_fields`, `check_and_fix_data`, `round_quantity` or `transaction_cost`, and order
  subclasses that override `pre_execute_check`, `post_complete` or `apply`, raise
  `NotImplementedError`. Use the Python engine for those, or `fill_price` and `cost` for
  a different fill or cost.
- A timezone-aware index works. Days are its local dates, and a local time that occurs
  twice (clocks going back) raises `ValueError`.
- Python strategies get their params unchanged, objects and lists included. Native C++
  strategies only take numbers, strings and bools, and other values raise `TypeError`
  naming the keys.

## Parameter sweeps

`runner.run_batch(params_list)` returns one result per parameter dict, in order. Python
strategies run in a process pool, so the strategy class must be importable by the
workers: define it in a module, not in a notebook or inside a function, and put the
calling code under `if __name__ == "__main__":`. Otherwise `run_batch` warns and runs
serially, as it does here. With `engine="cpp"` and only native C++ strategies, it uses
the C++ thread pool instead.

```python
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)  # serial: SMACross isn't importable
    sweep = runner.run_batch([{"fast": f, "slow": 30} for f in (5, 10, 20)])
final = [r.book_history.Main.total.iloc[-1] for r in sweep]
```

## Timing details

- **End-of-day tasks** (interest, and recording `book_history`) run once per date, after
  that date's last row. With intraday bars, `book_history` has one row per day.
- **Dates** are taken in the index's own timezone. Keep the index in the exchange's local
  time, naive or localized.
- **Custom assets** subclass `yabte.backtest.asset.Asset` and implement `data_fields()`,
  which returns `(field, flags)` pairs. The flags combine `ADFI_AVAILABLE_AT_OPEN`,
  `ADFI_AVAILABLE_AT_CLOSE` and `ADFI_REQUIRED`. They also implement
  `check_and_fix_data`, `intraday_traded_price` and `end_of_day_price`.

## Utilities

- `yabte.utilities.strategy_helpers.crossover(s1, s2)`: `True` if `s1` crossed above
  `s2` between the last two values.
- `yabte.utilities.portopt`:
  - `hierarchical_risk_parity.hrp(corr, sigma)`;
  - `inverse_volatility.inverse_volatility(cov)`;
  - `minimum_variance.minimum_variance(Sigma, mu, r)`, for a target return `r`.

  Each returns portfolio weights that sum to 1.
- `yabte.utilities.black_scholes.black_scholes(S, K, tau, sigma, r=0, call=True)`:
  European option `price`, `delta`, `gamma` and `vega`, vectorised over numpy arrays.
- `yabte.utilities.simulation`:
  - `geometric_brownian_motion.gbm_simulate_paths`;
  - `heston.heston_simulate_paths`;
  - `weiner.weiner_simulate_paths`.

  These generate synthetic prices.
- `import yabte.utilities.pandas_extension` registers `.prc` on DataFrames and Series:
  `returns`, `log_returns`, `capm_returns()`, `sharpe_ratio()`.
- `yabte.utilities.plot.plotly.strategy_runner.plot_strategy_runner_result(srr, runner)`
  plots a run; there is also a matplotlib version. `yabte.utilities.pyfolio` converts a
  result for pyfolio tear sheets.

```python
from yabte.utilities.portopt.inverse_volatility import inverse_volatility

closes = data.loc[:, (slice(None), "Close")].droplevel(1, axis=1)
weights = inverse_volatility(closes.pct_change().dropna().cov().to_numpy())
assert np.isclose(weights.sum(), 1)
```
