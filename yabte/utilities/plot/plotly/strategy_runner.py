from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ....backtest import StrategyRunner, StrategyRunnerResult


def plot_strategy_runner_result(
    srr: StrategyRunnerResult,
    sr: StrategyRunner,
    settings: dict[str, Any] | None = None,
):
    """Display the results of a strategy run using plotly.

    Plots a grid of charts with each column representing a book and rows representing
    each asset's price series along with long/short positioning and volumne series. A
    bottom row shows the value of each book as a price series.
    """
    default_settings: dict[str, Any] = {}

    if isinstance(settings, dict):
        default_settings = default_settings | settings
    s = pd.Series(default_settings, dtype=object)

    traded_assets = [
        a for a in srr.assets if a.name in srr.transaction_history.asset_name.unique()
    ]

    dpi = 100
    col_width = 8 * dpi
    row_unit_height = 3 * dpi
    ncols = len(srr.books)
    nrows = 1 + len(traded_assets)

    subplot_titles = [a.name for a in (traded_assets)] + ["Book Value"]
    row_heights = [10 for a in (traded_assets)] + [2]
    specs = [[{"secondary_y": True} for c in range(ncols)] for r in range(nrows)]

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
        start_cell="top-left",
        specs=specs,
        vertical_spacing=0.05,
    )

    for col, book in enumerate(srr.books, start=1):
        for row, asset in enumerate(traded_assets, start=1):
            prices = asset._filter_data(sr.data)

            fig.add_trace(
                go.Candlestick(
                    x=prices.index,
                    open=prices.Open,
                    high=prices.High,
                    low=prices.Low,
                    close=prices.Close,
                ),
                row=row,
                col=col,
                secondary_y=True,
            )

            fig.add_trace(
                go.Bar(
                    x=prices.index,
                    y=prices.Volume,
                    marker_color="lightgrey",
                ),
                row=row,
                col=col,
                secondary_y=False,
            )

            fig.update_xaxes(rangeslider_visible=False, row=row, col=col)
            fig.update_yaxes(
                title=asset.denom, secondary_y=True, showgrid=True, row=row, col=col
            )
            fig.update_yaxes(
                title="Volume", secondary_y=False, showgrid=False, row=row, col=col
            )

            # add some range selector buttons
            fig.update_xaxes(
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all"),
                    ]
                ),
                row=1,
                col=col,
            )

            trans_hist = srr.transaction_history.query(
                "asset_name==@asset.name and book==@book.name"
            )
            pos_hist = (
                trans_hist.groupby("ts")
                .agg(
                    quantity=("quantity", "sum"),
                    labels=(
                        "order_label",
                        lambda L: " ".join(l for l in L if l is not None),
                    ),
                )
                .reindex(prices.index)
            )

            shorts = prices[pos_hist.eval("quantity < 0")][["Low"]].join(
                pos_hist.labels
            )
            fig.add_trace(
                go.Scatter(
                    x=shorts.index,
                    y=shorts.Low,
                    customdata=shorts.labels,
                    mode="markers",
                    marker_symbol="arrow-down",
                    marker_color="red",
                    marker_size=10,
                    hovertemplate="%{x}<br>%{y}<br>%{customdata}<extra></extra>",
                ),
                row=row,
                col=col,
                secondary_y=True,
            )

            longs = prices[pos_hist.eval("quantity > 0")][["High"]].join(
                pos_hist.labels
            )
            fig.add_trace(
                go.Scatter(
                    x=longs.index,
                    y=longs.High,
                    customdata=shorts.labels,
                    mode="markers",
                    marker_symbol="arrow-up",
                    marker_color="green",
                    marker_size=10,
                    hovertemplate="%{x}<br>%{y}<br>%{customdata}<extra></extra>",
                ),
                row=row,
                col=col,
                secondary_y=True,
            )

    row = nrows
    bh = srr.book_history.loc[:, book.name]
    fig.add_trace(
        go.Scatter(
            x=bh.index,
            y=bh.total,
        ),
        row=row,
        col=col,
    )

    fig.update_xaxes(rangeslider_visible=True, row=row, col=col)
    fig.update_yaxes(title=book.denom, showgrid=True, row=row, col=col)
    fig.update_xaxes(title="Date", row=row, col=col)

    fig.update_layout(
        height=nrows * row_unit_height,
        width=ncols * col_width,
        showlegend=False,
        title_text="Strategy Runner Report",
    )

    return fig
