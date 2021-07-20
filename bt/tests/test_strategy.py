from functools import partial
import os

import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..strategy import Strategy, Signal
from .. import indicators

data = pd.Series(
    [np.random.randint(1, 100) for _ in range(100)], name="close"
).to_frame()


class ExampleStrategy(Strategy):
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


@pytest.fixture(scope="function")
def strategy():
    data = pd.Series(
        [np.random.randint(1, 100) for _ in range(100)], name="close"
    ).to_frame()
    strategy = ExampleStrategy(data)
    return strategy, data


def test_init():

    # Should work with a pandas.Series and a pandas.DataFrame
    strategy = ExampleStrategy(data)
    strategy = ExampleStrategy(data.close)

    # When given the start_index parameter as None, it should default to the first index
    # in the data
    strategy = ExampleStrategy(data, start_index=None)
    assert strategy._start_index == data.index[0]

    # When given the start_index parameter it must be the same type
    # as the index of the data
    with pytest.raises(TypeError):
        strategy = ExampleStrategy(data, start_index="This is a string")


def test_add_indicator(strategy):
    strat, data = strategy

    # Reset the indicators
    strat._indicator_functions = []

    # Should append the (name, function) tuple to the self._indicator_functions list
    sma = partial(indicators.sma, window=10)
    strat.add_indicator(indicator=sma, name="sma_short")

    assert strat._indicator_functions == [("sma_short", sma)]


def test_data(strategy):
    strat, data = strategy

    # The data property should show all the data if the strategy is not currently backtesting
    # Otherwise it will only show the data that is currently viewable to the backtest
    assert strat.data.equals(data)

    # When the strategy has viewable data, it should return the viewable data instead
    # of the full data
    strat = ExampleStrategy(data, start_index=10)

    # Line pulled from Strategy.backtest
    strat._viewable_data = strat._all_data.loc[: strat._start_index]

    assert strat.data.equals(strat._viewable_data)


def test_calculate_indicators(strategy):
    strat, data = strategy

    # Manually calculate the indicators (This is usually done by the backtest)
    strat._calculate_indicators()

    # Ensure that the indicators are calculated correctly
    values = indicators.sma(data["close"], window=20)
    assert strat._all_data.sma_short.equals(values)

    # Ensure that the indicators are named correctly
    assert strat.data.sma_short.name == "sma_short"

    # Ensure that the indicators are calculated correctly with a dataframe

    values = indicators.macd(data["close"])
    assert strat._all_data[values.columns].equals(values)


def test_get_signal():
    class ExampleStrategy(Strategy):
        def __init__(self, data: pd.Series, start_index=None) -> None:
            super().__init__(data, start_index)

        def is_buy(self) -> bool:
            return self.data.close.iloc[-1] > self.data.close.iloc[-2]

        def is_sell(self) -> bool:
            return self.data.close.iloc[-1] < self.data.close.iloc[-2]

    data = pd.Series([1, 2, 3])
    strategy = ExampleStrategy(data)
    assert strategy.get_signal() == Signal.BUY

    data = pd.Series([3, 2, 1])
    strategy = ExampleStrategy(data)
    assert strategy.get_signal() == Signal.SELL

    data = pd.Series([1, 1, 1])
    strategy = ExampleStrategy(data)
    assert strategy.get_signal() == Signal.HOLD


def test_backtest(strategy):
    # strategy = ExampleStrategy(data, start_index=30)
    strat, data = strategy

    results = strat.backtest(initial_balance=1000)

    # Check the results exist
    assert results

    # Check that the proper keys exist in the results
    keys = ["initial_balance", "balance", "trades", "portfolio_balance"]
    for key in keys:
        assert key in results

    # Check that the initial_balance is correct
    assert results["initial_balance"] == 1000

    # make sure that the len of porfolio_balance is equal to the len of the data
    # being backtested over
    assert len(results["portfolio_balance"]) == len(
        strat._all_data[strat._start_index :]
    )


def test_plot_results(strategy):
    strat, data = strategy

    # This should raise a ValueError if the strategy has not been backtested
    with pytest.raises(ValueError):
        strat.plot_results()

    results = strat.backtest(initial_balance=1000)

    # Check that the internal resutls exist
    assert strat._results

    # Should return a plotly figure
    fig = strat.plot_results()
    assert isinstance(fig, go.Figure)
