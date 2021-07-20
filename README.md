# BT Backtesting

a simple backtesting library


### Usage

```
from functools import partial

import pandas as pd
from bt import Strategy, indicators

class SMA(Strategy):
    """
    A Simple moving average strategy example
    """

    def __init__(self, data: pd.DataFrame, start_index=None) -> None:
        super().__init__(data, start_index)

        sma_short = partial(sma, window=10)
        self.add_indicator(sma_short, "sma_short")

        sma_long = partial(sma, window=30)
        self.add_indicator(sma_long, "sma_long")

    def is_buy(self) -> bool:
        return self.data.sma_short.iloc[-1] > self.data.sma_long.iloc[-1]

    def is_sell(self) -> bool:
        return self.data.sma_short.iloc[-1] < self.data.sma_long.iloc[-1]


if __name__ == "__main__":
    df = pd.read_csv(
        "../data/aapl.csv", index_col="date", parse_dates=True, infer_datetime_format=True
    ).drop(columns="Unnamed: 0")

    # Initializing the Strategy object with the data passed to it.
    strat = SMA(df)

    # Backtest using the previously passed data
    strat.backtest()

    strat.plot_results(
        "sma",
        show=False,
        plot_indicators=[["sma_short", "sma_long"]],
        plot_table=True,
        auto_open=True,
    )
```


# The Strategy

The strategy is an Abstract Base Class (ABC) that requires two methods to be defined within it's subclasses `is_buy()` and `is_sell()`. 

These two functions determine if, at any given point in time during the backtest, the strategy should buy or sell.

For example if one wanted to define a buy function to buy anytime the current price is higher than the previous tick's price you could do it like so
```
def is_buy(self):
    # self.data refers to the currenly available data when backtesting
    # or everything from the data passed in up to the current index
    # `data_passed_in[: current_index]`

    # Get the most recent price and the previous price
    current_price = self.data['close'].iloc[-1]
    prev_price = self.data['close'].iloc[-2]

    if current_price > prev_price:
        return True # Return True to indicate a buy signal

    return False # Return False to not indicate a buy signal
```

The `is_sell()` function works in a very similar fashion. Return `True` if you wish to sell, and `False` otherwise.

```
def is_sell(self):
    current_price = self.data['close'].iloc[-1]
    prev_price = self.data['close'].iloc[-2]

    if current_price > prev_price:
        return True # Return True to sell

    return False
```