import logging
import time
from typing import List

import numpy as np
import pandas as pd

# Configure logging to avoid debug spam during benchmark
logging.basicConfig(level=logging.INFO)

from yabte.backtest.asset import OHLCAsset
from yabte.backtest.order import Order, SimpleOrder
from yabte.backtest.strategy import Strategy
from yabte.backtest.strategyrunner import StrategyRunner
from yabte.utilities.simulation.geometric_brownian_motion import gbm_simulate_paths

try:
    from yabte.backtest._yabte_backtest_lib import SMAXOStrategy as CppSMAXOStrategy

    HAS_CPP = True
except ImportError:
    HAS_CPP = False
    print("Warning: C++ extension not found.")

# --- 1. Generate Synthetic Data ---


def generate_synthetic_data(n_assets=50, n_years=5, freq="D"):
    """Generates synthetic OHLCV data for multiple assets using GBM."""
    print(f"Generating synthetic data for {n_assets} assets over {n_years} years...")

    # Time parameters
    if freq == "D":
        n_steps = n_years * 252
        dt = 1 / 252
    else:
        # Simplified for now, assuming daily
        n_steps = n_years * 252
        dt = 1 / 252

    T = n_years

    # Random drift and volatility for each asset
    np.random.seed(42)  # Ensure reproducibility
    mu = np.random.normal(0.05, 0.02, n_assets)  # 5% annual drift avg
    sigma = np.random.normal(0.2, 0.05, n_assets)  # 20% annual vol avg
    S0 = 100

    # Correlation matrix (identity for simplicity, uncorrelated assets)
    R = np.eye(n_assets)

    # Use 1 simulation path
    prices = gbm_simulate_paths(
        S0=S0, mu=mu, sigma=sigma, R=R, T=T, n_steps=n_steps, n_sims=1
    )
    # prices shape: (n_steps, 1, n_assets)

    # Flatten to (n_steps, n_assets)
    prices = prices[:, 0, :]

    # Create DataFrame
    dates = pd.date_range(start="2020-01-01", periods=n_steps, freq="B")
    asset_names = [f"ASSET_{i}" for i in range(n_assets)]

    # Create MultiIndex DataFrame (Asset, Field)
    data_frames = {}

    for i, asset in enumerate(asset_names):
        close_series = prices[:, i]
        # Synthetic variations
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

    # Concatenate to get (Date, (Asset, Field))
    # concat(dict) creates keys as first level of columns
    df_combined = pd.concat(data_frames, axis=1)

    return df_combined, asset_names


# --- 2. Define Python Strategy ---


class PythonSMAXOComparable(Strategy):
    def __init__(self, asset_name, n, m):
        self.asset_name = asset_name
        self.n = n
        self.m = m

    def on_close(self):
        # Access data via standard API which includes slicing overhead
        df_hist = self.data

        # Check sufficient history
        if len(df_hist) < self.m:
            return

        try:
            # MultiIndex column access
            closes = df_hist[(self.asset_name, "Close")]
        except KeyError:
            return

        # Calculate SMAs using pandas
        # This mirrors the computation in C++ (summing/averaging)
        # but with Python/Pandas overhead
        sma_short = closes.iloc[-self.n :].mean()
        sma_long = closes.iloc[-self.m :].mean()

        if sma_short > sma_long:
            self.orders.append(SimpleOrder(asset_name=self.asset_name, size=1))
        elif sma_short < sma_long:
            self.orders.append(SimpleOrder(asset_name=self.asset_name, size=-1))


# --- 3. Benchmark Execution ---


def run_benchmark():
    n_assets = 50
    n_years = 5
    n_sma = 50
    m_sma = 200

    data, asset_names = generate_synthetic_data(n_assets=n_assets, n_years=n_years)

    # Use OHLCAsset because it implements check_and_fix_data
    assets = [OHLCAsset(name=n) for n in asset_names]

    print(f"Data shape: {data.shape}")
    print(f"Number of timesteps: {len(data)}")

    # --- Python Run ---
    print("\nStarting Python Engine Benchmark (Comparable SMAXO Strategy)...")
    # Using the comparable SMAXO strategy
    strategies_py = [PythonSMAXOComparable(name, n_sma, m_sma) for name in asset_names]

    runner_py = StrategyRunner(
        data=data, assets=assets, strategies=strategies_py, engine="python"
    )

    start_py = time.perf_counter()
    runner_py.run()
    end_py = time.perf_counter()

    time_py = end_py - start_py
    print(f"Python Engine Time: {time_py:.4f} seconds")

    # --- C++ Run ---
    if HAS_CPP:
        print("\nStarting C++ Engine Benchmark (Pure C++ SMAXO)...")
        # Instantiate the C++ strategy class directly
        strategies_cpp = [CppSMAXOStrategy(name, n_sma, m_sma) for name in asset_names]

        runner_cpp = StrategyRunner(
            data=data, assets=assets, strategies=strategies_cpp, engine="cpp"
        )

        try:
            start_cpp = time.perf_counter()
            runner_cpp.run()
            end_cpp = time.perf_counter()

            time_cpp = end_cpp - start_cpp
            print(f"C++ Engine Time:    {time_cpp:.4f} seconds")

            speedup = time_py / time_cpp
            print(f"\nSpeedup Factor:     {speedup:.2f}x")

        except Exception as e:
            print(f"\nC++ Engine failed with error: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("Skipping C++ benchmark.")


if __name__ == "__main__":
    run_benchmark()
