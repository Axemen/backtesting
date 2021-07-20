from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Union

import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from .util import pct_change


class Signal(Enum):
    BUY = 1
    HOLD = 0
    SELL = -1


class Strategy(ABC):
    """
    Define a Strategy base class that will be inherited from to define
    backtesting strategies.
    """

    def __init__(self, data: Union[pd.Series, pd.DataFrame], start_index=None):
        """
        :param data: pandas.Series | pandas.DataFrame
            The data to use when backtesting the strategy.
            (If a Series is given, it will be converted to a DataFrame
            with one column of the name `close`)

        :param start_index: int
            The index to start backtesting from.
            (Previous data will be used for calculating indicators)
            (Must match the index of the data)

        :return: Strategy
            The strategy instance.
        """
        if isinstance(data, pd.Series):
            data = data.to_frame(name="close")

        assert "close" in data.columns, "data must have a 'close' column"

        self._all_data = data

        if start_index is None:
            self._start_index = self._all_data.index[0]
        else:
            # Ensure the start index is valid
            if type(start_index) != type(data.index[0]):
                raise TypeError(
                    "start_index type must be the same as the index of the data param"
                )

            self._start_index = start_index

        # TODO change to a dictionary
        self._indicator_functions = []
        self._viewable_data = None

    @property
    def data(self) -> pd.DataFrame:
        """
        Returns the data of the strategy to be used in backtesting
        """
        if getattr(self, "_viewable_data") is None:
            return self._all_data

        return self._viewable_data

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

    def add_indicator(self, indicator: function, name: str = None) -> None:
        """
        Add an indicator to the strategy

        the given function should take in the Strategy.data DataFrame and return a pandas.Series or pandas.DataFrame.
        the returned series will be added as a column to the Strategy.data as the column `name`.

        :param indicator: function
            The indicator function.

        :param name: str
            The name of the indicator column in the data.
            If not given, the name/columns of the series/dataframe will be used.
            name is ignored for dataframe indicators.
        """

        # TODO change _indicator_functions to a dictionary with the key being a prefix to add
        # before the column names of the indicators if the indicator fn returns a dataframe
        self._indicator_functions.append((name, indicator))

    def _calculate_indicators(self) -> None:
        """
        Calculate indicators and add them as columns to the strategy._all_data DataFrame.
        """
        # TODO adjust to work with a dictionary of indicators instead
        for indicator_name, indicator in self._indicator_functions:

            result = indicator(self._all_data)

            if isinstance(result, pd.Series):  # if indicator returns a series
                if indicator_name is None:
                    indicator_name = result.name
                    assert (
                        indicator_name not in self._all_data.columns
                    ), "Indicator name already exists"
                self._all_data[indicator_name] = result

            elif isinstance(result, pd.DataFrame):
                indicator_name = result.columns

                for name in result.columns:
                    if name in self._all_data.columns:
                        raise ValueError(f"Indicator name already exists: {name}")

                self._all_data[indicator_name] = result.values

            else:
                raise ValueError(
                    "Indicator must return a pandas.Series or pandas.DataFrame"
                )

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

        :return: dict
        """

        # TODO add way to specify how much to buy or sell
        self._results = {
            "initial_balance": initial_balance,
            "balance": initial_balance,
            # List of trade spans (start_index, end_index, profit)
            "trades": [],
            "portfolio_balance": [],  # (index, balance)
        }

        # Calculate the indicators on the initial data
        self._calculate_indicators()

        balance = initial_balance
        is_trade_open = False
        open_trade = {"index": None, "price": None, "num_shares": 0}

        self._viewable_data = self._all_data.loc[: self._start_index]

        for index, row in tqdm(
            self._all_data.loc[self._start_index :].iterrows(),
            disable=not verbose,
            total=len(self._all_data.loc[self._start_index :]),
        ):
            if self._start_index == index:
                # No visible data if they're the same
                self._results["portfolio_balance"].append((index, balance))
                continue
            self._viewable_data = self._all_data[self._start_index : index]
            # Calculate the signals
            signal = self.get_signal()

            if not is_trade_open:
                if signal == Signal.BUY:
                    is_trade_open = True
                    open_trade["index"] = index
                    open_trade["price"] = row["close"]
                    open_trade["num_shares"] = int(balance / row["close"])
                    balance -= open_trade["num_shares"] * row["close"]

            elif is_trade_open:
                if signal == Signal.SELL:
                    is_trade_open = False
                    self._results["trades"].append(
                        (open_trade["index"], index, row["close"] - open_trade["price"])
                    )
                    balance += open_trade["num_shares"] * row["close"]
                    open_trade = {}

            # Update the portfolio balance
            if is_trade_open:
                self._results["portfolio_balance"].append(
                    (index, balance + (row["close"] * open_trade["num_shares"]))
                )
            else:
                self._results["portfolio_balance"].append((index, balance))

        # Sell at end of data
        if is_trade_open:
            is_trade_open = False
            self._results["trades"].append(
                (open_trade["index"], index, row["close"] - open_trade["price"])
            )
            balance += open_trade["num_shares"] * row["close"]
            open_trade = {}

        self._results["balance"] = balance
        return self._results

    def plot_results(
        self,
        filename: str = None,
        plot_indicators=[],
        plot_table=False,
        auto_open=False,
        show=False,
    ) -> go.Figure:
        """
        Plot the results of the backtest.

        :param filename: str
            The filename to save the plot to. If None, the plot will not be saved.

        :param plot_indicators: bool
            Whether to plot the indicators.
            ex: [['indicator_1', 'indicator_2'], ['indicator_3', 'indicator_4']]
            1 and 2 will be plot on the same figure, 3 and 4 on the next figure.

        :param plot_table: list[list[str]]
            Whether to plot the results table. (Plotly only)

        :param auto_open: bool
            Whether to open the generated HTML file in a browser.

        :param show: bool
            Whether to display the plot in a Jupyter notebook.
            Will open a webserver if running in a script

        :return: None
        """
        # Ensure there are backtest results
        if not getattr(self, "_results", False):
            raise ValueError("Backtesting must be run before plotting results")

        num_rows = sum(
            [
                2,  # Default plots
                len(plot_indicators),
            ]
        )

        specs = [[{"type": "xy"}] for _ in range(num_rows)]
        if plot_table:
            specs.append([{"type": "table"}])
            num_rows += 1

        fig = make_subplots(
            rows=num_rows,
            cols=1,
            shared_xaxes=False,
            subplot_titles=(
                "Portfolio Balance",
                "Closing Price w/ Trades overlayed",
            ),
            specs=specs,
        )

        # Plot the portfolio balance
        portfolio_balance = go.Scatter(
            x=[x[0] for x in self._results["portfolio_balance"]],
            y=[x[1] for x in self._results["portfolio_balance"]],
            name="Balance",
        )

        fig.add_trace(portfolio_balance, 1, 1)

        # Plot the trades and the close price
        closes = self._viewable_data["close"]
        close_trace = go.Scatter(x=closes.index, y=closes.values, name="Close")
        fig.add_trace(close_trace, 2, 1)

        trades = self._results["trades"]
        for trade in trades:
            if trade[2] >= 0:
                fig.add_vrect(
                    x0=trade[0],
                    x1=trade[1],
                    fillcolor="green",
                    opacity=0.25,
                    layer="below",
                    line_width=0,
                    annotation_text="Profit",
                    annotation_position="top left",
                    row=2,
                    col=1,
                )

            else:
                fig.add_vrect(
                    x0=trade[0],
                    x1=trade[1],
                    fillcolor="red",
                    opacity=0.25,
                    layer="below",
                    line_width=0,
                    annotation_text="Loss",
                    annotation_position="top left",
                    row=2,
                    col=1,
                )

        # Plot the indicators
        for row, indicator_set in enumerate(plot_indicators, start=3):
            for indicator_name in indicator_set:
                data = self._viewable_data[indicator_name]
                fig.add_trace(
                    go.Scatter(x=data.index, y=data.values, name=indicator_name),
                    row,
                    1,
                )

        # Create table showing summary statistics
        if plot_table:
            table = go.Table(
                header=dict(values=["Statistic", "Value"]),
                cells=dict(
                    values=[
                        [
                            "Number of Trades made",
                            "Initial Balance",
                            "Final Balance",
                            "Profit/Loss",
                            "Buy/Hold %",
                            "Profit/Loss %",
                        ],
                        [
                            len(trades),  # Number of Trades
                            # Initial Balance
                            self._results["initial_balance"],
                            self._results["balance"],  # Final balance
                            # Profit/loss
                            (
                                self._results["balance"]
                                - self._results["initial_balance"]
                            ),
                            pct_change(closes.iloc[0], closes.iloc[-1]),  # Buy/Hold %
                            pct_change(
                                self._results["initial_balance"],
                                self._results["balance"],
                            ),  # Profit/Loss %
                        ],
                    ]
                ),
            )
            fig.add_trace(table, num_rows, 1)

        fig["layout"].update(title="Backtest Results")

        if filename:
            fig.write_html(f"{filename}.html", auto_open=auto_open)

        if show:
            fig.show()

        return fig
