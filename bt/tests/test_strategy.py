from functools import partial

import pytest
import numpy as np
import pandas as pd

from ..strategy import Strategy, Signal
from .. import indicators

data = pd.Series(
    [np.random.randint(1, 100) for _ in range(100)], name="close"
).to_frame()


class TestStrategy(Strategy):
    """
    A Simple moving average strategy example
    """

    def __init__(self, data: pd.Series, start_index=None) -> None:
        super().__init__(data, start_index)

        sma_short = partial(indicators.sma, window=20, col="close")
        self.add_indicator(sma_short, name="sma_short")

        sma_long = partial(indicators.sma, window=50, col="close")
        self.add_indicator(sma_long, name="sma_long")

        macd = partial(indicators.macd, col="close")
        self.add_indicator(macd, name="macd")

    def is_buy(self) -> bool:
        return self.data.sma_short.iloc[-1] > self.data.sma_long.iloc[-1]

    def is_sell(self) -> bool:
        return self.data.sma_short.iloc[-1] < self.data.sma_long.iloc[-1]


def test_init():

    # Should work with a pandas.Series and a pandas.DataFrame
    strategy = TestStrategy(data)
    strategy = TestStrategy(data.close)

    # When given the start_index parameter as None, it should default to the first index
    # in the data
    strategy = TestStrategy(data, start_index=None)
    assert strategy._start_index == data.index[0]

    # When given the start_index parameter it must be the same type
    # as the index of the data
    with pytest.raises(TypeError):
        strategy = TestStrategy(data, start_index="This is a string")


def test_add_indicator():
    strategy = TestStrategy(data)

    # Reset the indicators
    strategy._indicator_functions = []

    # Should append the (name, function) tuple to the self._indicator_functions list
    sma = partial(indicators.sma, window=10)
    strategy.add_indicator(indicator=sma, name="sma_short")

    assert strategy._indicator_functions == [("sma_short", sma)]


def test_data():
    strategy = TestStrategy(data)

    # The data property should show all the data if the strategy is not currently backtesting
    # Otherwise it will only show the data that is currently viewable to the backtest
    assert strategy.data.equals(data)

    # When the strategy has viewable data, it should return the viewable data instead
    # of the full data
    strategy = TestStrategy(data, start_index=10)

    # Line pulled from Strategy.backtest
    strategy._viewable_data = strategy._all_data.loc[: strategy._start_index]

    assert strategy.data.equals(strategy._viewable_data)


def test_calculate_indicators():
    strategy = TestStrategy(data)

    # Manually calculate the indicators (This is usually done by the backtest)
    strategy._calculate_indicators()

    # Ensure that the indicators are calculated correctly
    values = indicators.sma(data["close"], window=20)
    assert strategy._all_data.sma_short.equals(values)

    # Ensure that the indicators are named correctly
    assert strategy.data.sma_short.name == "sma_short"

    # Ensure that the indicators are calculated correctly with a dataframe

    values = indicators.macd(data["close"])
    assert strategy._all_data[[values.columns]].equals(values)
