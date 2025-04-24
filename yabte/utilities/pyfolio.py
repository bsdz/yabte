import pandas as pd

from ..backtest import StrategyRunnerResult


def extract_rets_pos_txn_from_yabte(
    srr: StrategyRunnerResult,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Extract returns, positions, transactions and leverage from the
    backtest data structure returned by yabte.backtest.StrategyRunner.run().

    The returned data structures are in a format compatible with the
    rest of pyfolio and can be directly passed to
    e.g. tears.create_full_tear_sheet().

    Example:
    ---------------------------------------------
    >>> srr = my_strategy_runner.run()
    >>> returns, positions, transactions =
    >>>     pyfolio.utils.extract_rets_pos_txn_from_yabte(srr)
    >>> pyfolio.tears.create_full_tear_sheet(returns,
    >>>     positions, transactions)
    """

    returns = srr.books[0].history.total.pct_change().dropna()

    transactions = (
        srr.transaction_history[["ts", "quantity", "price", "asset_name"]]
        .set_index("ts")
        .rename(columns=dict(quantity="amount", asset_name="symbol"))
    )
    transactions["amount"] = transactions["amount"].astype(float)
    transactions["price"] = transactions["price"].astype(float)
    transactions.index = transactions.index.tz_localize("UTC")

    th2 = srr.transaction_history.copy()
    th2["date"] = th2.ts.dt.date
    th2["quantity"] = th2.quantity.astype(float)
    th2["price"] = th2.price.astype(float)
    th2["total"] = th2.total.astype(float)
    cum_pos = (
        th2.groupby(["date", "asset_name"]).quantity.sum().unstack().cumsum().fillna(0)
    )
    price_mean = th2.groupby(["date", "asset_name"]).price.mean().unstack().fillna(0)
    pos_no_cash = cum_pos * price_mean

    pos_cash = srr.book_history.loc[:, (slice(None), "cash")].sum(axis=1).rename("cash")

    positions = pd.concat(
        [pos_no_cash, pos_cash.reindex(pos_no_cash.index, method="ffill")], axis=1
    )
    positions.index = pd.to_datetime(positions.index).tz_localize("UTC")

    return returns, positions, transactions
