from functools import partial

import pandas as pd

from bt import Strategy

def sma(data: pd.DataFrame, window: int) -> pd.Series:
    """
    Simple moving average
    """
    return data['close'].rolling(window).mean()

class SMA(Strategy):
    """
    A Simple moving average strategy example
    """
    def __init__(self, data: pd.Series, start_index=None) -> None:
        super().__init__(data, start_index)

        sma_short = partial(sma, window=10)
        sma_long = partial(sma, window=20)

        self.add_indicator("sma_short", sma_short)
        self.add_indicator("sma_long", sma_long)

    def is_buy(self) -> bool:
        return self.data.sma_short.iloc[-1] > self.data.sma_long.iloc[-1]

    def is_sell(self) -> bool:
        return self.data.sma_short.iloc[-1] < self.data.sma_long.iloc[-1]



df = pd.read_csv("data/aapl.csv")
# print(df.head())
SMA(df).backtest()