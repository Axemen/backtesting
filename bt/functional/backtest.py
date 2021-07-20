from typing import Callable

import numpy as np
import pandas as pd


def backtest(
    data: pd.DataFrame,
    is_buy: Callable,
    is_sell: Callable,
    start_index=None,
    initial_balance=1000,
) -> dict:
    """
    Backtest a strategy.

    :param data: DataFrame of price data and indicators.
    :param is_buy: Function that returns True if a given row should be a buy.
    :param is_sell: Function that returns True if a given row should be a sell.
    :param start_index: Index of the first row to be used for backtesting.
    :return: Dictionary of backtest results.
    """
    if start_index is None:
        start_index = data.index[0]

    # Initialize the backtest dictionary
    results = {
        "initial_balance": initial_balance,
        "balance": initial_balance,
        "trades": [],
        "portfolio_balance": [],
    }

    open_trade = {}

    # Iterate over the rows of the data
    for index, row in data.iloc[start_index:].iterrows():

        viewable_data = data.iloc[start_index:index]

        if len(viewable_data) == 0:
            continue

        if is_buy(viewable_data) and not open_trade:
            open_trade = {
                "index": index,
                "price": row["close"],
                "num_shares": int(initial_balance / row["close"]),
            }
            results["balance"] -= open_trade["num_shares"] * row["close"]

        elif is_sell(viewable_data) and open_trade:
            results["trades"].append(
                (open_trade["index"], index, row["close"] - open_trade["price"])
            )
            results["balance"] += open_trade["num_shares"] * row["close"]
            open_trade = {}

        if open_trade:
            results["portfolio_balance"].append(
                (index, results["balance"] + row["close"] * open_trade["num_shares"])
            )
        else:
            results["portfolio_balance"].append((index, results["balance"]))

    # Sell at end of data
    if open_trade:
        results["trades"].append(
            (open_trade["index"], index, row["close"] - open_trade["price"])
        )
        results["balance"] += open_trade["num_shares"] * row["close"]
        open_trade = {}

    return results
