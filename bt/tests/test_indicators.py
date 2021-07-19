import numpy as np
import pandas as pd
import pytest

from ..indicators import ema, macd, sma

data = pd.Series(np.random.randn(100), name="close")


def test_sma():

    # sma should be a moving average
    sma_ = sma(data)

    # sma should be a pandas series
    assert isinstance(sma_, pd.Series), "sma result should be a pandas series"

    # sma should be the same length as the input
    assert len(sma_) == len(
        data
    ), "len of sma result should be the same as the initial data"

    assert sma(
        data.to_frame(), col="close"
    ).any(), "the sma should accept a pandas dataframe for the data param"

    # sma should raise a ValueError if data is a pd.DataFrame and col is not specified
    with pytest.raises(ValueError):
        sma(data.to_frame())

    # sma should raise an error if the input is not a pandas series or pandas dataframe
    with pytest.raises(TypeError):
        sma(1)


def test_ema():

    # ema should be a moving average
    ema_ = ema(data)

    # ema should be a pandas series
    assert isinstance(ema_, pd.Series), "ema result should be a pandas series"

    # ema should be the same length as the input
    assert len(ema_) == len(
        data
    ), "len of ema result should be the same as the initial data"

    assert ema(
        data.to_frame(), col="close"
    ).any(), "the ema should accept a pandas dataframe for the data param"

    # ema should raise a ValueError if data is a pd.DataFrame and col is not specified
    with pytest.raises(ValueError):
        ema(data.to_frame())

    # ema should raise an error if the input is not a pandas series or pandas dataframe
    with pytest.raises(TypeError):
        ema(1)


def test_macd():
    # calculate the macd
    macd_ = macd(data)

    # macd should be a pandas dataframe
    assert isinstance(macd_, pd.DataFrame), "macd result should be pd.DataFrame"

    # macd should be the same length as the input
    assert len(macd_) == len(
        data
    ), "len of macd result should be the same as the initial data"

    assert not macd(
        data.to_frame(), col="close"
    ).empty, "the macd should accept a pandas dataframe for the data param"

    # macd should raise a ValueError if data is a pd.DataFrame and col is not specified
    with pytest.raises(ValueError):
        macd(data.to_frame())

    # macd should raise an error if the input is not a pandas series or pandas dataframe
    with pytest.raises(TypeError):
        macd(1)
