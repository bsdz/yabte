import pandas as pd


def crossover(series1: pd.Series, series2: pd.Series) -> bool:
    """Determine if two series cross over one another. Returns `True` if `series1` just
    crosses above `series2`.

        >>> crossover(self.data.Close, self.sma)
        True
    """
    try:
        return (
            series1.iloc[-2] < series2.iloc[-2] and series1.iloc[-1] > series2.iloc[-1]
        )
    except IndexError:
        return False
