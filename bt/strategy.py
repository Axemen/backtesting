from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd
import numpy as np
from tqdm import tqdm


class Signal(Enum):
    BUY = 1
    HOLD = 0
    SELL = -1


class Strategy(ABC):
    """
    Define a Strategy base class that will be inherited from to define backtesting strategies.
    """

    def __init__(self, data: pd.DataFrame, start_index=0):
        """
        :param data: pandas.Series
            The data to use when backtesting the strategy.
        :param start_index: int
            The index to start backtesting from. (Previous data will be used for calculating indicators)

        :return: Strategy
            The strategy instance.
        """
        assert 'close' in data.columns, "data must have a 'close' column"

        self._all_data = data
        self._start_index = start_index

        self._indicator_functions = {}
        self._backtesting = False

    @property
    def data(self) -> pd.DataFrame:
        if self._backtesting:
            return self._viewable_data
        else:
            return self._all_data

    @abstractmethod
    def is_buy(self) -> bool:
        """
        Return True to buy.
        """
        raise NotImplementedError("Should implement is_buy()")

    @abstractmethod
    def is_sell(self) -> bool:
        """
        Return True to sell.
        """
        raise NotImplementedError("Should implement is_sell()")

    def add_indicator(self, name: str, indicator: function) -> None:
        """
        Add an indicator to the strategy

        To be used in __init__ function of subclasses

        :param name: str
            The name of the indicator

        :param indicator: function
            The indicator function. Should take in the data and return a pandas.Series.
            the calculated values will be added as a column to the Strategy.data with the column `name`.
        """
        self._indicator_functions[name] = indicator

    def _calculate_indicators(self) -> None:
        """
        Calculate indicators

        :param close: float
            The close price of the current bar.
        """
        for indicator_name, indicator in self._indicator_functions.items():
            self._all_data[indicator_name] = indicator(self._all_data)

    def _update(self, close, index) -> None:
        """
        Update indicators

        :param close: float
            The close price of the current bar.
        """
        self.data.loc[index, 'close'] = close
        self._calculate_indicators()

    def get_signal(self) -> Signal:
        """
        Return the signal of the strategy
        """
        if self.is_buy():
            return Signal.BUY
        elif self.is_sell():
            return Signal.SELL
        else:
            return Signal.HOLD

    def backtest(self, initial_balance=1000, verbose=True) -> dict:
        """
        Backtest the strategy.

        :return: BacktestResult
        """

        # TODO add way to specify how much to buy or sell
        self._backtesting = True
        self._results = {
            'initial_balance': initial_balance,
            'balance': initial_balance,

            # List of trade spans (start_index, end_index, profit)
            'trades': [],
            'portfolio_balance': [],  # (index, balance)
        }

        # Calculate the indicators on the initial data
        self._calculate_indicators()

        balance = initial_balance
        is_trade_open = False
        open_trade = {
            'index': None,
            'price': None,
            'num_shares': 0
        }

        self._viewable_data = pd.DataFrame(columns=self._all_data.columns)

        for index, row in tqdm(self._all_data.iterrows(), disable=not verbose, total=len(self._all_data)):
            self._viewable_data.loc[index] = row
            # Calculate the signals
            signal = self.get_signal()

            if not is_trade_open:
                if signal == Signal.BUY:
                    is_trade_open = True
                    open_trade['index'] = index
                    open_trade['price'] = row['close']
                    open_trade['num_shares'] = int(balance / row['close'])
                    balance -= open_trade['num_shares'] * row['close']

            elif is_trade_open:
                if signal == Signal.SELL:
                    is_trade_open = False
                    self._results['trades'].append((open_trade['index'], index,
                                                    open_trade['price'] - row['close']))
                    balance += open_trade['num_shares'] * row['close']
                    open_trade = {}

            # Update the portfolio balance
            if is_trade_open:
                self._results['portfolio_balance'].append((index,
                                                           balance + row['close'] * open_trade['num_shares']))
            else:
                self._results['portfolio_balance'].append((index, balance))

        self._backtesting = False
        return self._results