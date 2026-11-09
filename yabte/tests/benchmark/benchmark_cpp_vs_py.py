"""Compare the Python and C++ engines on the same strategy.

Every scenario runs one SMA crossover strategy (buy 100 when the short SMA crosses
above the long, sell 100 when it crosses below) and the trades are checked to be
identical across scenarios, so the timings compare the same work:

- ``python/pandas``: Python engine, idiomatic strategy reading ``self.data`` each step
- ``cpp/pandas``: C++ engine, same strategy
- ``python/numpy``: Python engine, strategy with near-zero cost (precomputed numpy
  signals), so the time is mostly the engine
- ``cpp/numpy``: C++ engine, same
- ``cpp/native``: C++ engine, native C++ ``SMAXOStrategy``

Usage (C++ extension on PYTHONPATH for a dev build)::

    PYTHONPATH=cpp/build/Release/pybind python -m yabte.tests.benchmark.benchmark_cpp_vs_py
    python -m yabte.tests.benchmark.benchmark_cpp_vs_py --assets 10,50 --years 5 --repeats 3
"""

import argparse
import statistics
import time
from decimal import Decimal

import numpy as np
import pandas as pd

from yabte.backtest import Book, OHLCAsset, SimpleOrder, Strategy, StrategyRunner
from yabte.backtest.runner_cpp import yabte_cpp

SMA_SHORT, SMA_LONG = 50, 200


def generate_data(n_assets: int, n_years: int, seed: int = 42):
    """Synthetic daily OHLCV data for `n_assets` assets over `n_years` years."""
    rng = np.random.default_rng(seed)
    steps = n_years * 252
    index = pd.date_range("2015-01-01", periods=steps, freq="B")
    names = [f"A{i}" for i in range(n_assets)]
    cols = {}
    for name in names:
        close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.0126, steps)))
        open_ = close * (1 + rng.normal(0, 0.002, steps))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, steps)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, steps)))
        volume = rng.integers(1000, 100000, steps).astype(float)
        for field, values in zip(
            ["Open", "High", "Low", "Close", "Volume"],
            [open_, high, low, close, volume],
        ):
            cols[(name, field)] = values
    df = pd.DataFrame(cols, index=index)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df, names


class PandasSMAXO(Strategy):
    """Idiomatic: SMAs added to ``self.data`` in init(), read back each step."""

    def __init__(self, names):
        self.names = list(names)

    def init(self):
        close = self.data.loc[:, (slice(None), "Close")]
        short = close.rolling(SMA_SHORT).mean().rename({"Close": "S"}, axis=1, level=1)
        long = close.rolling(SMA_LONG).mean().rename({"Close": "L"}, axis=1, level=1)
        self.data = pd.concat([self.data, short, long], axis=1).sort_index(axis=1)

    def on_close(self):
        data = self.data
        if len(data) < SMA_LONG + 2:
            return
        for name in self.names:
            s = data[(name, "S")].iloc[-2:].values
            l = data[(name, "L")].iloc[-2:].values
            if s[0] < l[0] and s[1] > l[1]:
                self.orders.append(SimpleOrder(asset_name=name, size=100))
            elif l[0] < s[0] and l[1] > s[1]:
                self.orders.append(SimpleOrder(asset_name=name, size=-100))


class NumpySMAXO(Strategy):
    """Near-zero strategy cost: signals precomputed with numpy in init()."""

    def __init__(self, names):
        self.names = list(names)

    def init(self):
        self.i = -1
        close = self.data.loc[:, [(name, "Close") for name in self.names]]
        self.s = close.rolling(SMA_SHORT).mean().values
        self.l = close.rolling(SMA_LONG).mean().values

    def on_close(self):
        self.i += 1
        i = self.i
        if i < SMA_LONG + 1:
            return
        s0, s1, l0, l1 = self.s[i - 1], self.s[i], self.l[i - 1], self.l[i]
        crossed = ((s0 < l0) & (s1 > l1)) | ((l0 < s0) & (l1 > s1))
        for j in np.flatnonzero(crossed):
            size = 100 if s1[j] > l1[j] else -100
            self.orders.append(SimpleOrder(asset_name=self.names[j], size=size))


SCENARIOS = [
    ("python", "pandas"),
    ("cpp", "pandas"),
    ("python", "numpy"),
    ("cpp", "numpy"),
    ("cpp", "native"),
]


def make_strategies(kind: str, names):
    if kind == "native":
        return [yabte_cpp.SMAXOStrategy(name, SMA_SHORT, SMA_LONG) for name in names]
    return [{"pandas": PandasSMAXO, "numpy": NumpySMAXO}[kind](names)]


def run_scenario(data, names, engine: str, kind: str):
    """Run once; returns (seconds, result).

    Timing includes runner set-up.
    """
    start = time.perf_counter()
    result = StrategyRunner(
        data=data,
        assets=[OHLCAsset(name=name) for name in names],
        strategies=make_strategies(kind, names),
        books=[Book(name="Main", cash=Decimal(100_000_000))],
        engine=engine,
    ).run()
    return time.perf_counter() - start, result


def canonical_trades(result) -> pd.DataFrame:
    th = result.transaction_history
    return (
        pd.DataFrame(
            {
                "ts": pd.to_datetime(th["ts"]).values,
                "asset": th["asset_name"].values,
                "quantity": th["quantity"].astype(float).values,
                "price": th["price"].astype(float).values,
            }
        )
        .sort_values(["ts", "asset"])
        .reset_index(drop=True)
    )


def same_trades(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return (
        len(a) == len(b)
        and (a.ts.values == b.ts.values).all()
        and (a.asset.values == b.asset.values).all()
        and np.allclose(a.quantity, b.quantity)
        and np.allclose(a.price, b.price)
    )


def run_benchmark(n_assets: int, n_years: int, repeats: int):
    data, names = generate_data(n_assets, n_years)
    print(
        f"\n{n_assets} assets, {n_years} years ({len(data)} steps), {repeats} repeat(s)"
    )

    reference = None
    for engine, kind in SCENARIOS:
        label = f"{engine}/{kind}"
        if engine == "cpp" and yabte_cpp is None:
            print(f"  {label:14s} skipped: C++ extension not found")
            continue
        times = []
        for _ in range(repeats):
            seconds, result = run_scenario(data, names, engine, kind)
            times.append(seconds)
        trades = canonical_trades(result)
        if reference is None:
            reference = trades
        match = "same trades" if same_trades(reference, trades) else "TRADES DIFFER"
        print(
            f"  {label:14s} median {statistics.median(times):9.4f}s"
            f"  (min {min(times):.4f}s)  {len(trades)} trades, {match}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--assets", default="10,50", help="comma-separated counts")
    parser.add_argument("--years", default="5", help="comma-separated years")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    for n_years in map(int, args.years.split(",")):
        for n_assets in map(int, args.assets.split(",")):
            run_benchmark(n_assets, n_years, args.repeats)


if __name__ == "__main__":
    main()
