import pandas as pd
import numpy as np
import pytest

from ..functional.backtest import backtest


def test_backtest():
    data = pd.DataFrame(
        {
            "close": np.random.random(100),
        }
    )

    def is_buy(data: pd.DataFrame):
        if len(data) < 2:
            return False

        if data["close"].iloc[-1] > data["close"].iloc[-2]:
            return True

    def is_sell(data: pd.DataFrame):
        if len(data) < 2:
            return False

        if data["close"].iloc[-1] < data["close"].iloc[-2]:
            return True

    # Adding indicator
    results = backtest(data, is_buy, is_sell)
    assert isinstance(results, dict)
