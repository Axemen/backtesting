from functools import partial
import sys

import pandas as pd

from bt import Strategy


def sma(data: pd.DataFrame, window: int) -> pd.Series:
    """
    Simple moving average
    """
    return data["close"].rolling(window).mean()


def macd(
    data: pd.DataFrame, short_window=12, long_window=26, signal_window=9
) -> pd.Series:
    """
    Moving average convergence/divergence
    """
    short_sma = data["close"].ewm(span=short_window).mean()
    long_sma = data["close"].ewm(span=long_window).mean()
    macd = short_sma - long_sma
    signal = macd.ewm(span=signal_window).mean()
    return pd.DataFrame(macd, name="macd")


class SMA(Strategy):
    """
    A Simple moving average strategy example
    """

    def __init__(self, data: pd.Series, start_index=None) -> None:
        super().__init__(data, start_index)

        sma_short = partial(sma, window=10)
        self.add_indicator("sma_short", sma_short)

    def is_buy(self) -> bool:
        return self.data.sma_short.iloc[-1] > self.data.sma_long.iloc[-1]

    def is_sell(self) -> bool:
        return self.data.sma_short.iloc[-1] < self.data.sma_long.iloc[-1]


df = pd.read_csv(
    "data/aapl.csv", index_col="date", parse_dates=True, infer_datetime_format=True
).drop(columns="Unnamed: 0")
# print(df.head())
strat = SMA(df)
strat.backtest()
strat.plot_results(
    "sma", backend="plotly", show=True, plot_indicators=[["sma_short", "sma_long"]]
)
