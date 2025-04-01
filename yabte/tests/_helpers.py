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


def generate_ohlc_dataset(
    mu: float,
    T: float,
    n_steps: int,
    S0: list[float],
    vol: list[float],
    names: list[str],
    start,
    freq,
    R: np.ndarray | None = None,
    rng=None,
):
    if R is None:
        R = np.identity(len(names))

    vol = np.array(vol)
    S0 = np.array(S0)

    p = gbm_simulate_paths(
        S0=S0, mu=mu, sigma=vol, R=R, T=T, n_steps=n_steps, n_sims=1, rng=rng
    )

    ix = pd.date_range(start, freq=freq, periods=n_steps)
    df = pd.DataFrame(p[:, 0, :], index=ix, columns=names)

    ohlc_df = df.resample("D").ohlc()

    assets = [OHLCAsset(name=n, denom="USD") for n in names]

    return assets, ohlc_df.rename(
        dict(open="Open", high="High", low="Low", close="Close"), axis=1
    )


if __name__ == "__main__":
    # 20 days in 30 min samples; 1 day = 48 x 30 min
    # T/N = 20/365 / 20 / 48 = 1/365/48

    data = generate_ohlc_dataset(
        mu=0.05,
        T=20 / 365,
        n_steps=20 * 48,
        S0=[10, 100, 1000, 50],
        vol=[0.2, 0.3, 0.1, 0.4],
        names=["A1 Inc", "B2 Corp", "C3 Ltd", "D4 Inc"],
        start=datetime(2025, 1, 1, 0, 0),
        freq="30min",
        rng=np.random.default_rng(12345),
    )

    print(data)
