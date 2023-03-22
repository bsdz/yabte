from pathlib import Path

import pandas as pd

from yabte.backtest.asset import Asset

data_dir = Path(__file__).parent / "data"
notebooks_dir = Path(__file__).parents[2] / "notebooks"


def generate_nasdaq_dataset():
    assets = []
    dfs = []
    for csv_pth in (data_dir / "nasdaq").glob("*.csv"):
        name = csv_pth.stem
        assets.append(Asset(name=name, denom="USD"))
        df = pd.read_csv(csv_pth, index_col=0, parse_dates=[0])
        df.columns = pd.MultiIndex.from_tuples([(name, f) for f in df.columns])
        dfs.append(df)

    return assets, pd.concat(dfs, axis=1)
