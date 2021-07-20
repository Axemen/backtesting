from functools import partial
import sys

sys.path.append("..")

import pandas as pd

from bt import Strategy


def sma(data: pd.DataFrame, window: int) -> pd.Series:
    """
    Simple moving average
    """
    return data["close"].rolling(window).mean()


class SMA(Strategy):
    """
    A Simple moving average strategy example
    """

    def __init__(self, data: pd.Series, start_index=None) -> None:
        super().__init__(data, start_index)

        sma_short = partial(sma, window=10)
        self.add_indicator(sma_short, "sma_short")

        sma_long = partial(sma, window=30)
        self.add_indicator(sma_long, "sma_long")

    def is_buy(self) -> bool:
        return self.data.sma_short.iloc[-1] > self.data.sma_long.iloc[-1]

    def is_sell(self) -> bool:
        return self.data.sma_short.iloc[-1] < self.data.sma_long.iloc[-1]


df = pd.read_csv(
    "../data/aapl.csv", index_col="date", parse_dates=True, infer_datetime_format=True
).drop(columns="Unnamed: 0")

strat = SMA(df)
strat.backtest()
strat.plot_results(
    "sma",
    backend="plotly",
    show=False,
    plot_indicators=[["sma_short", "sma_long"]],
    plot_table=True,
    auto_open=True,
)
