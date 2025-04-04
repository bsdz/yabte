from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from yabte.backtest.asset import OHLCAsset
from yabte.utilities.simulation.geometric_brownian_motion import gbm_simulate_paths

data_dir = Path(__file__).parent / "data"
notebooks_dir = Path(__file__).parents[2] / "notebooks"


def generate_nasdaq_dataset():
    assets = []
    dfs = []
    for csv_pth in (data_dir / "nasdaq").glob("*.csv"):
        name = csv_pth.stem
        assets.append(OHLCAsset(name=name, denom="USD"))
        df = pd.read_csv(csv_pth, index_col=0, parse_dates=[0])
        df.columns = pd.MultiIndex.from_tuples([(name, f) for f in df.columns])
        dfs.append(df)

    return assets, pd.concat(dfs, axis=1)


def generate_gbm_dataset(
    S0: list[float],
    mu: float,
    vol: list[float],
    names: list[str],
    index: pd.DatetimeIndex,
    R: np.ndarray | None = None,
    rng=None,
):
    # NOTE: vol is already in annualized form
    if R is None:
        R = np.identity(len(names))

    vol = np.array(vol)
    S0 = np.array(S0)

    n_steps = len(index)
    T = (index.max() - index.min()).days / 365
    p = gbm_simulate_paths(
        S0=S0, mu=mu, sigma=vol, R=R, T=T, n_steps=n_steps, n_sims=1, rng=rng
    )

    df = pd.DataFrame(p[:, 0, :], index=index, columns=names)

    return df


def generate_ohlc_dataset(
    S0: list[float],
    mu: float,
    vol: list[float],
    names: list[str],
    start,
    end,
    freq,
    tick_freq="1min",
    R: np.ndarray | None = None,
    rng=None,
):
    if R is None:
        R = np.identity(len(names))

    vol = np.array(vol)
    S0 = np.array(S0)

    tick_index = pd.date_range(start, end, freq=tick_freq)
    df = generate_gbm_dataset(
        S0=S0,
        mu=mu,
        vol=vol,
        names=names,
        index=tick_index,
        rng=rng,
    )

    ohlc_df = df.resample(freq).ohlc()

    assets = [OHLCAsset(name=n, denom="USD") for n in names]

    return assets, ohlc_df.rename(
        dict(open="Open", high="High", low="Low", close="Close"), axis=1
    )


if __name__ == "__main__":

    data = generate_gbm_dataset(
        S0=[10, 100, 1000, 50],
        mu=0.05,
        vol=[0.2, 0.3, 0.1, 0.4],
        names=["A1 Inc", "B2 Corp", "C3 Ltd", "D4 Inc"],
        index=pd.date_range(datetime(2025, 1, 1), periods=20 * 48, freq="30min"),
        rng=np.random.default_rng(12345),
    )

    assets, data = generate_ohlc_dataset(
        S0=[10, 100, 1000, 50],
        mu=0.05,
        vol=[0.2, 0.3, 0.1, 0.4],
        names=["A1 Inc", "B2 Corp", "C3 Ltd", "D4 Inc"],
        start=datetime(2025, 1, 1),
        end=datetime(2025, 3, 1),
        freq="30min",
        tick_freq="1min",
        rng=np.random.default_rng(12345),
    )

    print(data)
