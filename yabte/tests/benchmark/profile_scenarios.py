import cProfile
import io
import logging
import pstats
import time

import numpy as np
import pandas as pd

from yabte.backtest.asset import OHLCAsset
from yabte.backtest.order import SimpleOrder
from yabte.backtest.strategy import Strategy
from yabte.backtest.strategyrunner import StrategyRunner
from yabte.utilities.simulation.geometric_brownian_motion import gbm_simulate_paths

# Configure logging
logging.basicConfig(level=logging.WARNING)

try:
    import yabte.backtest._yabte_backtest_lib as cpp_lib
    from yabte.backtest._yabte_backtest_lib import SMAXOStrategy as CppSMAXOStrategy

    print(f"DEBUG: Loaded C++ extension from {cpp_lib.__file__}")

    HAS_CPP = True
except ImportError:
    HAS_CPP = False
    print("Warning: C++ extension not found.")

# --- 1. Generate Synthetic Data ---


def generate_synthetic_data(n_assets=50, n_years=5):
    """Generates synthetic OHLCV data."""
    print(f"Generating synthetic data for {n_assets} assets over {n_years} years...")
    n_steps = n_years * 252
    T = n_years

    np.random.seed(42)
    mu = np.random.normal(0.05, 0.02, n_assets)
    sigma = np.random.normal(0.2, 0.05, n_assets)
    S0 = 100
    R = np.eye(n_assets)

    prices = gbm_simulate_paths(
        S0=S0, mu=mu, sigma=sigma, R=R, T=T, n_steps=n_steps, n_sims=1
    )
    prices = prices[:, 0, :]  # Flatten

    dates = pd.date_range(start="2020-01-01", periods=n_steps, freq="B")
    asset_names = [f"ASSET_{i}" for i in range(n_assets)]

    data_frames = {}
    for i, asset in enumerate(asset_names):
        close_series = prices[:, i]
        high_series = close_series * (1 + np.abs(np.random.normal(0, 0.005, n_steps)))
        low_series = close_series * (1 - np.abs(np.random.normal(0, 0.005, n_steps)))
        open_series = close_series * (1 + np.random.normal(0, 0.002, n_steps))
        volume_series = np.random.randint(1000, 100000, n_steps)

        df_asset = pd.DataFrame(
            {
                "Close": close_series,
                "High": high_series,
                "Low": low_series,
                "Open": open_series,
                "Volume": volume_series,
            },
            index=dates,
        )
        data_frames[asset] = df_asset

    return pd.concat(data_frames, axis=1), asset_names


# --- 2. Python Strategy ---


class PythonSMAXOComparable(Strategy):
    def __init__(self, asset_name, n, m):
        self.asset_name = asset_name
        self.n = n
        self.m = m

    def on_close(self):
        df_hist = self.data
        if len(df_hist) < self.m:
            return

        try:
            closes = df_hist[(self.asset_name, "Close")]
        except KeyError:
            return

        sma_short = closes.iloc[-self.n :].mean()
        sma_long = closes.iloc[-self.m :].mean()

        if sma_short > sma_long:
            self.orders.append(SimpleOrder(asset_name=self.asset_name, size=1))
        elif sma_short < sma_long:
            self.orders.append(SimpleOrder(asset_name=self.asset_name, size=-1))


# --- 3. Profiling Wrapper ---


class Profiler:
    def __init__(self, name):
        self.name = name
        self.metrics = {}

    def measure(self, label, func, *args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        duration = end - start
        self.metrics[label] = duration
        return result


def print_comparison_table(results):
    """Prints a comparison table of timings.

    results: dict of {scenario_name: {label: duration}}
    """
    if not results:
        return

    scenarios = list(results.keys())
    if not scenarios:
        return

    # Assume all scenarios have the same labels (sections)
    labels = list(results[scenarios[0]].keys())

    # Header
    # 20 chars for Section, then 30 chars for each scenario
    header = f"{'Section':<20} | " + " | ".join(f"{s:<30}" for s in scenarios)
    separator = "-" * len(header)

    print("\n" + separator)
    print(header)
    print(separator)

    # Rows
    for label in labels:
        row = f"{label:<20} | "
        for scenario in scenarios:
            duration = results[scenario].get(label, 0.0)
            # Just print the number, padding logic was a bit weird in thought process
            row += f"{duration:<30.4f} | "
        print(row)

    print(separator)


# --- 4. Main Execution ---


def run_profiles():
    # Production parameters
    n_assets = 50
    n_years = 5
    n_sma = 50
    m_sma = 200

    print(f"Profiling with n_assets={n_assets}, n_years={n_years}")

    data, asset_names = generate_synthetic_data(n_assets, n_years)
    assets = [OHLCAsset(name=n) for n in asset_names]

    all_results = {}

    # --- Scenario 1: Python Engine + Python Strategy ---
    print("\nRunning Scenario 1: Python Engine + Python Strategy")
    prof_py_py = Profiler("Python Engine + Python Strategy")

    strategies_py = [PythonSMAXOComparable(name, n_sma, m_sma) for name in asset_names]

    runner_py_py = prof_py_py.measure(
        "Data Loading",
        lambda: StrategyRunner(data=data, assets=assets, strategies=strategies_py),
    )
    prof_py_py.measure("Strategy Run", runner_py_py.run)

    all_results["Python Engine + Python Strategy"] = prof_py_py.metrics

    # --- Scenario 2: C++ Engine + Python Strategy ---
    print("\nRunning Scenario 2: C++ Engine + Python Strategy")
    prof_cpp_py = Profiler("C++ Engine + Python Strategy")

    if HAS_CPP:
        from yabte.backtest.runner_cpp import CppStrategyRunner

        # Re-instantiate python strategies to avoid state carry-over
        strategies_py_2 = [
            PythonSMAXOComparable(name, n_sma, m_sma) for name in asset_names
        ]

        runner_cpp_py = prof_cpp_py.measure(
            "Data Loading",
            lambda: CppStrategyRunner(
                data=data, assets=assets, strategies=strategies_py_2
            ),
        )
        prof_cpp_py.measure("Strategy Run", runner_cpp_py.run)

        all_results["C++ Engine + Python Strategy"] = prof_cpp_py.metrics

    # --- Scenario 3: C++ Engine + C++ Strategy (ONLY) ---
    print("\nRunning Scenario 3: C++ Engine + C++ Strategy")
    prof_cpp_cpp = Profiler("C++ Engine + C++ Strategy")

    if not HAS_CPP:
        print("Skipping C++ scenarios.")
        return

    from yabte.backtest.runner_cpp import CppStrategyRunner

    strategies_cpp = [CppSMAXOStrategy(name, n_sma, m_sma) for name in asset_names]

    runner_cpp_cpp = prof_cpp_cpp.measure(
        "Data Loading",
        lambda: CppStrategyRunner(data=data, assets=assets, strategies=strategies_cpp),
    )
    prof_cpp_cpp.measure("Strategy Run", runner_cpp_cpp.run)

    all_results["C++ Engine + C++ Strategy"] = prof_cpp_cpp.metrics

    # --- Print Final Table ---
    print_comparison_table(all_results)


if __name__ == "__main__":
    run_profiles()
