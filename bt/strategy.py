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
        assert "close" in data.columns, "data must have a 'close' column"

        self._all_data = data

        if start_index is None:
            self._start_index = self._all_data.index[0]
        else:
            self._start_index = start_index

        self._indicator_functions = []
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
        """
        self._indicator_functions.append((name, indicator))

    def _calculate_indicators(self) -> None:
        """
        Calculate indicators and add them as columns to the strategy._all_data DataFrame.
        """
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
                if indicator_name is None:
                    indicator_name = result.columns

                    for indicator in result.columns:
                        assert (
                            indicator not in self._all_data.columns
                        ), "Indicator name already exists"

                    self._all_data[indicator_name] = result.values

            else:
                raise ValueError(
                    "Indicator must return a pandas.Series or pandas.DataFrame"
                )

    def _update(self, close, index) -> None:
        """
        Update indicators

        :param close: float
            The close price of the current bar.
        """
        self.data.loc[index, "close"] = close
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

        :return: dict
        """

        # TODO add way to specify how much to buy or sell
        self._backtesting = True
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

        self._viewable_data = self._all_data.loc[: self._start_index].copy()

        for index, row in tqdm(
            self._all_data.loc[self._start_index :].iterrows(),
            disable=not verbose,
            total=len(self._all_data),
        ):
            self._viewable_data.loc[index] = row
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

        self._backtesting = False
        self._results["balance"] = balance
        return self._results

    def plot_results(
        self,
        filename: str = None,
        show=False,
        backend="plotly",
        plot_indicators=[],
        plot_table=False,
    ) -> None:
        """
        Plot the results of the backtest.

        :param filename: str
            The filename to save the plot to. If None, the plot will not be saved.

        :param show: bool
            Whether to show the plot.

        :param backend: str
            The backend to use. Valid options are 'matplotlib' and 'plotly'.

        :param plot_indicators: bool
            Whether to plot the indicators.

        :param plot_table: list[list[str]]
            Whether to plot the results table. (Plotly only)
            ex: [['indicator_1', 'indicator_2'], ['indicator_3', 'indicator_4']]
            1 and 2 will be plot on the same figure, 3 and 4 on the next figure.

        :return: None
        """
        # Ensure there are backtest results
        assert getattr(
            self, "_results"
        ), "Backtesting must be run before plotting results"

        if backend == "matplotlib":
            import matplotlib.pyplot as plt

            num_rows = sum(
                [
                    2,  # Default plots
                    len(self._indicator_functions) if plot_indicators else 0,
                ]
            )

            fig, axs = plt.subplots(num_rows, 1, sharex=False, figsize=(15, 10))

            fig.tight_layout(h_pad=5)

            # Plot the portfolio balance
            axs[0].set_title("Portfolio Balance")
            portfolio_balance = pd.Series(
                [x[1] for x in self._results["portfolio_balance"]],
                index=[x[0] for x in self._results["portfolio_balance"]],
                name="Balance",
            )
            portfolio_balance.plot(ax=axs[0])

            # Plot the trades and the close price
            axs[1].set_title("Closing Price")
            closes = self._viewable_data["close"]
            closes.plot(ax=axs[1])

            trades = self._results["trades"]
            axs[1].vlines(
                x=[x[0] for x in trades], ymin=min(closes), ymax=max(closes), color="b"
            )
            axs[1].vlines(
                x=[x[1] for x in trades], ymin=min(closes), ymax=max(closes), color="r"
            )

            for trade in trades:
                if trade[2] >= 0:
                    axs[1].axvspan(trade[0], trade[1], color="g", alpha=0.1)
                else:
                    axs[1].axvspan(trade[0], trade[1], color="r", alpha=0.1)

            # Plot the indicators
            if plot_indicators:
                for i, indicator_name in enumerate(self._indicator_functions, start=2):
                    data = self._viewable_data[indicator_name]
                    axs[i].set_title(indicator_name)
                    data.plot(ax=axs[i])

            if filename:
                plt.savefig(f"{filename}.png")
            if show:
                plt.show()

        elif backend == "plotly":
            import plotly.graph_objs as go
            from plotly.subplots import make_subplots

            num_rows = sum(
                [
                    2,  # Default plots
                    len(plot_indicators),
                    plot_table,
                ]
            )

            specs = [[{"type": "xy"}] for _ in range(num_rows)]
            if plot_table:
                specs.append([{"type": "table"}])

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

            fig.append_trace(portfolio_balance, 1, 1)

            # Plot the trades and the close price
            closes = self._viewable_data["close"]
            close_trace = go.Scatter(x=closes.index, y=closes.values, name="Close")
            fig.append_trace(close_trace, 2, 1)

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
                        annotation_text="Buy",
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
                        annotation_text="Sell",
                        annotation_position="top left",
                        row=2,
                        col=1,
                    )

            # Plot the indicators
            for row, indicator_set in enumerate(plot_indicators, start=3):
                for indicator_name in indicator_set:
                    data = self._viewable_data[indicator_name]
                    fig.append_trace(
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
                                (closes.iloc[-1] - closes.iloc[0])
                                / closes.iloc[0]
                                * 100,  # Buy/Hold %
                                (
                                    self._results["balance"]
                                    - self._results["initial_balance"]
                                )
                                / self._results["initial_balance"]
                                * 100,  # Profit/Loss %
                            ],
                        ]
                    ),
                )
                fig.append_trace(table, len(num_rows), 1)

            fig["layout"].update(title="Backtest Results")

            if show:
                fig.show()

            if filename:
                fig.write_html(f"{filename}.html", auto_open=True)
