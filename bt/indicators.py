import pandas as pd
from typing import Union


def sma(
    data: Union[pd.DataFrame, pd.Series], window: int = 10, col: str = None
) -> pd.Series:
    """
    :param data: [pd.DataFrame, pd.Series]
    :param window: int
    :param col: str

    :return: pandas.Series
    """

    if window <= 0:  # check if window is positive
        raise ValueError("window must be positive")

    if isinstance(data, pd.DataFrame):
        if col is None:
            raise ValueError("col must be specified if data is a pd.DataFrame")
        return data[col].rolling(window=window).mean()
    elif isinstance(data, pd.Series):
        return data.rolling(window=window).mean()
    else:
        raise TypeError("data must be pandas.DataFrame or pandas.Series")


def ema(
    data: Union[pd.DataFrame, pd.Series], window: int = 10, col: str = None
) -> pd.Series:
    """
    :param data: [pd.DataFrame, pd.Series]
    :param window: int
    :param col: str

    :return: pandas.Series
    """
    if isinstance(data, pd.DataFrame):
        if col is None:
            raise ValueError("col must be specified if data is a pd.DataFrame")
        return data[col].ewm(span=window).mean()
    elif isinstance(data, pd.Series):
        return data.ewm(span=window).mean()
    else:
        raise TypeError("data must be pandas.DataFrame or pandas.Series")


def macd(
    data: Union[pd.DataFrame, pd.Series],
    col: str = None,
    fast_window: int = 12,
    slow_window: int = 26,
    signal_window: int = 9,
) -> pd.DataFrame:
    """
    :param data: [pd.DataFrame, pd.Series]
    :param fast_window: int
    :param slow_window: int
    :param col: str

    :return: pandas.DataFrame
    """

    if isinstance(data, pd.DataFrame):
        if col is None:
            raise ValueError("col must be specified if data is a pd.DataFrame")
        data = data[col]
    elif not isinstance(data, pd.Series):
        raise TypeError("data must be pandas.DataFrame or pandas.Series")

    ema_fast = ema(data, fast_window)
    ema_slow = ema(data, slow_window)
    macd = ema_fast - ema_slow
    signal = ema(macd, signal_window, "macd")
    hist = macd - signal
    return pd.DataFrame({"macd": macd, "signal": signal, "hist": hist})
