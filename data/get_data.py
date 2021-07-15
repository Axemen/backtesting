import os
from urllib import parse

import pandas as pd
import requests as r
from dotenv import load_dotenv

load_dotenv()

IEX_TOKEN = os.getenv("IEX_TOKEN")

params = {"token": IEX_TOKEN}

BASE_URL = "https://cloud.iexapis.com/stable/"


symbol = "aapl"
timeframe = "5y"
endpoint = f"/stock/{symbol}/chart/{timeframe}?"
# Get the data

url = BASE_URL + endpoint + parse.urlencode(params)
print(url)
data = r.get(url).json()

pd.DataFrame(data).to_csv("aapl.csv")
