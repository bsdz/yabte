import pandas as pd


def crossover(series1: pd.Series, series2: pd.Series) -> bool:
    """
    Return `True` if `series1` just crossed over (above)
    `series2`.

        >>> crossover(self.data.Close, self.sma)
        True
    """
    try:
        return series1[-2] < series2[-2] and series1[-1] > series2[-1]
    except IndexError:
        return False