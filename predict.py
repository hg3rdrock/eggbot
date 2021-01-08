import ccxt
import numpy as np
import pandas as pd
from stable_baselines3 import A2C
from stockstats import StockDataFrame as Sdf

# Load data from binance API
# binance_cli = Client()
# klines = binance_cli.get_historical_klines('BTCUSDT', '1d', '120 days ago UTC')
# data = pd.DataFrame(klines,
#                     dtype=np.float,
#                     columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av',
#                              'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
# data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
# data.set_index('timestamp', inplace=True)
# data.drop(columns=['close_time', 'quote_av', 'tb_base_av', 'tb_quote_av', 'ignore'], inplace=True)
# data.rename(columns={'timestamp': 'date'}, inplace=True)

# Load data from ccxt
exchange = ccxt.binance()
klines = exchange.fetchOHLCV('BTC/USDT', '1d', limit=120)
data = pd.DataFrame(klines, dtype=np.float, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
data.set_index('timestamp', inplace=True)
data.rename(columns={'timestamp': 'date'}, inplace=True)

# Add tech indicators
df = Sdf.retype(data)
tech_indicator_list = ['macd', 'rsi_30', 'cci_30', 'dx_30', 'kdjk', 'open_2_sma', 'boll', 'close_10.0_le_5_c', 'wr_10',
                       'dma', 'trix']
for indicator in tech_indicator_list:
    _ = df[indicator]

# Add column: daily_return
df['daily_return'] = df.close.pct_change(1)


# Make an observation from the most recent day's data
def make_obs(day_loc):
    dp = df.iloc[day_loc]
    # print(dp)
    obs = [10000] + [dp.close] + [0] + \
          sum([[dp[ti]] for ti in tech_indicator_list], []) + \
          [dp.open, dp.high, dp.low, dp.daily_return]
    return obs


# Create model
model = A2C.load('./trained_models/A2C_20201224-20h24.zip')

# Make a prediction
for i in range(-10, 0):
    obs = make_obs(i)
    action, _ = model.predict(obs)
    print(f"price: {obs[1]}, action: {action}")
