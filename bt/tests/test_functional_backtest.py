import pandas as pd
import pytest

from ..functional.backtest import backtest


def test_backtest():
    data = pd.DataFrame(
        {
            "close": [1, 2, 3, 4, 5],
        }
    )

    # Adding indicator
    data["sma"] = data["close"] > 2
