from functools import partial

import pytest
import numpy as np
import pandas as pd

from ..strategy import Strategy
from .. import indicators

data = pd.Series(np.random.randn(100), name="close").to_frame()


class TestStrategy(Strategy):
    """
    A Simple moving average strategy example
    """

    def __init__(self, data: pd.Series, start_index=None) -> None:
        super().__init__(data, start_index)

        sma = partial(indicators.sma, window=20)
        self.add_indicator(sma, name="sma_short")

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
