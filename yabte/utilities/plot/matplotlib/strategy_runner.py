from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FuncFormatter

from ....backtest import StrategyRunner
from .marker_updater import MarkerUpdater


def plot_strategy_runner(sr: StrategyRunner, settings: dict[str, Any] | None = None):
    """Display the results of a strategy run using matplotlib.

    Plots a grid of charts with each column representing a book and rows
    representing each asset's price series along with long/short
    positioning and volumne series. A bottom row shows the value of each
    book as a price series.
    """
    default_settings = {
        "candle_body_width": 0.8,
        "candle_wick_width": 0.2,
        "candle_up_color": "green",
        "candle_down_color": "red",
        "date_format": "%d-%b-%Y",
        "long_markers": {},
        "short_markers": {},
    }

    if isinstance(settings, dict):
        default_settings = default_settings | settings
    s = pd.Series(default_settings, dtype=object)

    traded_assets = [
        a for a in sr.assets if a.name in sr.transaction_history.asset_name.unique()
    ]

    col_width = 8
    row_unit_height = 3
    ncols = len(sr.books)
    nrows = 1 + 2 * len(traded_assets)
    hr = [3, 1] * len(traded_assets) + [2]
    fig, axss = plt.subplots(
        nrows,
        ncols,
        height_ratios=hr,
        sharex=True,
        squeeze=False,
        figsize=(ncols * col_width, nrows * row_unit_height),
    )
    fig.suptitle("Strategy Runner Report")

    marker_updater = MarkerUpdater()
    date_formatter = DateFormatter(s.date_format)

    for book, axs in zip(sr.books, axss.T):
        for i, asset in enumerate(traded_assets):
            prices = sr.data[asset.data_label]

            up = prices[prices.Close >= prices.Open]
            down = prices[prices.Close < prices.Open]

            pax = axs[2 * i]
            pax.set_title(asset.name)
            pax.set_ylabel(asset.denom)
            pax.fmt_xdata = date_formatter

            pax.bar(
                up.index,
                up.Close - up.Open,
                s.candle_body_width,
                bottom=up.Open,
                color=s.candle_up_color,
            )
            pax.bar(
                up.index,
                up.High - up.Close,
                s.candle_wick_width,
                bottom=up.Close,
                color=s.candle_up_color,
            )
            pax.bar(
                up.index,
                up.Low - up.Open,
                s.candle_wick_width,
                bottom=up.Open,
                color=s.candle_up_color,
            )

            pax.bar(
                down.index,
                down.Close - down.Open,
                s.candle_body_width,
                bottom=down.Open,
                color=s.candle_down_color,
            )
            pax.bar(
                down.index,
                down.High - down.Open,
                s.candle_wick_width,
                bottom=down.Open,
                color=s.candle_down_color,
            )
            pax.bar(
                down.index,
                down.Low - down.Close,
                s.candle_wick_width,
                bottom=down.Close,
                color=s.candle_down_color,
            )

            vax = axs[2 * i + 1]
            vax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x/1000:,.0f}"))
            vax.set_ylabel(f"Volume (thousands)")
            vax.fmt_xdata = date_formatter

            vax.bar(
                prices.index,
                prices.Volume,
                color="black",
            )

            trans_hist = sr.transaction_history.query(
                "asset_name==@asset.name and book==@book.name"
            )
            pos_hist = (
                trans_hist.groupby("ts")
                .agg(
                    quantity=("quantity", np.sum),
                    labels=(
                        "order_label",
                        lambda L: " ".join(l for l in L if l is not None),
                    ),
                )
                .reindex(prices.index)
            )

            for l in pos_hist.labels.dropna().unique():
                marker_long = s.long_markers.get("", "^")
                marker_short = s.long_markers.get("", "v")

                prices.Low[pos_hist.eval("quantity < 0 and labels==@l")].rename(
                    f"Short {l}".strip()
                ).plot(
                    color="red",
                    marker=marker_short,
                    markersize=6,
                    linestyle="None",
                    ax=pax,
                )
                prices.High[pos_hist.eval("quantity > 0 and labels==@l")].rename(
                    f"Long {l}".strip()
                ).plot(
                    color="green",
                    marker=marker_long,
                    markersize=6,
                    linestyle="None",
                    ax=pax,
                )

            pax.legend()
            marker_updater.add_ax(pax, ["size"])

        bax = axs[-1]
        bax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x/1000:,.0f}"))
        bax.set_ylabel(f"{book.denom} (thousands)")

        sr.book_history.loc[:, book.name].total.plot(ax=bax)
        bax.set_xlabel("Date")
        bax.fmt_xdata = date_formatter

    # markers don't scale with zoom, so update them in this event hook
    fig.tight_layout()
    return fig, axs
