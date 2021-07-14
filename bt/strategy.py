from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd
from tqdm import tqdm


class Signal(Enum):
    BUY = 1
    HOLD = 0
    SELL = -1


class Strategy(ABC):
    """
    Define a Strategy base class that will be inherited from to define backtesting strategies.
    """

    def __init__(self, data: pd.DataFrame, start_index=None):
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

        if start_index is None:
            self._start_index = self._all_data.iloc[0].index
        else:
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

        self._viewable_data = self._all_data[:self._start_index]

        for index, row in tqdm(self._all_data[self._start_index:].iterrows(), disable=not verbose, total=len(self._all_data)):
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
                                                    row['close'] - open_trade['price']))
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

    def plot_results(self, filename: str = None, show=False, backend='matplotlib'):
        """
        Plot the results of the backtest.
        """
        # Ensure there are backtest results
        assert getattr(
            self, "_results"), "Backtesting must be run before plotting results"

        if backend == 'matplotlib':
            import matplotlib.pyplot as plt

            fig, (ax, ax2) = plt.subplots(2, 1, sharex=False, figsize=(15, 10))

            # Plot the portfolio balance
            ax.set_title("Portfolio Balance")
            portfolio_balance = pd.Series([x[1] for x in self._results['portfolio_balance']], index=[
                                          x[0] for x in self._results['portfolio_balance']], name='Balance')
            portfolio_balance.plot(ax=ax)

            # Plot the trades and the close price
            ax2.set_title("Closing Price")
            closes = self._viewable_data['close']
            closes.plot(ax=ax2)

            trades = self._results['trades']
            ax2.vlines(x=[x[0] for x in trades], ymin=min(
                closes), ymax=max(closes), color='b')
            ax2.vlines(x=[x[1] for x in trades], ymin=min(
                closes), ymax=max(closes), color='r')

            for trade in trades:
                if trade[2] >= 0:
                    ax2.axvspan(trade[0], trade[1], color='g', alpha=0.1)
                else:
                    ax2.axvspan(trade[0], trade[1], color='r', alpha=0.1)

            if filename:
                plt.savefig(f'{filename}.png')
            if show:
                plt.show()

        elif backend == 'plotly':
            import plotly.graph_objs as go
            from plotly.subplots import make_subplots

            fig = make_subplots(rows=3, cols=1, shared_xaxes=False,
                                subplot_titles=(
                                    'Portfolio Balance', 'Closing Price w/ Trades overlayed'),
                                specs=[
                                    [{'type': 'xy'}],
                                    [{'type': 'xy'}],
                                    [{'type': 'table'}]
                                ]
                                )

            # Plot the portfolio balance
            portfolio_balance = go.Scatter(x=[x[0] for x in self._results['portfolio_balance']],
                                           y=[x[1]
                                               for x in self._results['portfolio_balance']],
                                           name='Balance')

            fig.append_trace(portfolio_balance, 1, 1)

            # Plot the trades and the close price
            closes = self._viewable_data['close']
            close_trace = go.Scatter(
                x=closes.index, y=closes.values, name='Close')
            fig.append_trace(close_trace, 2, 1)

            trades = self._results['trades']
            for trade in trades:
                if trade[2] >= 0:
                    fig.add_vrect(
                        x0=trade[0], x1=trade[1],
                        fillcolor="green", opacity=0.25,
                        layer="below", line_width=0,
                        annotation_text="Buy", annotation_position="top left",
                        row=2, col=1
                    )

                else:
                    fig.add_vrect(
                        x0=trade[0], x1=trade[1],
                        fillcolor="red", opacity=0.25,
                        layer="below", line_width=0,
                        annotation_text="Sell", annotation_position="top left",
                        row=2, col=1
                    )

            # Create table showing summary statistics
            table = go.Table(
                header=dict(values=['Statistic', 'Value']),
                cells=dict(values=[['Number of Trades', len(trades)],
                                   ['Initial Balance',
                                       self._results['initial_balance']],
                                   ['Final Balance', self._results['balance']],
                                   ['Profit %', (self._results['balance'] - self._results['initial_balance']
                                                 ) / self._results['initial_balance'] * 100],
                                   ['Buy and Hold %', closes.iloc[-1] -
                                       closes.iloc[0] / closes.iloc[0] * 100]
                                   ['Average Trade %', sum(
                                       [x[2] for x in trades]) / len(trades) * 100],
                                   ],

                           )
            )
            fig.append_trace(table, 3, 1)

            if filename:
                fig.write_image(f'{filename}.png')
            if show:
                fig.show()

            fig['layout'].update(title='Backtest Results')

            if show:
                fig.show()

            if filename:
                fig.write_html(f'{filename}.html', auto_open=True)
