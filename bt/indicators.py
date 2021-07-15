# Common technical indicators
def sma(data, period):
    return data.rolling(window=period).mean()