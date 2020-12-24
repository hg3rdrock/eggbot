import time
import os

import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from env import SinglePairEnv
from stockstats import StockDataFrame as Sdf
from finrl.config import config

# Read data from local csv file
data = pd.read_csv(os.path.join("data", "BTCUSDT-1d-data.csv"))
data.drop(columns=['close_time', 'quote_av', 'tb_base_av', 'tb_quote_av', 'ignore'], inplace=True)
data.rename(columns={'timestamp': 'date'}, inplace=True)

# Add tech indicator columns
df = Sdf.retype(data)
tech_indicator_list = config.TECHNICAL_INDICATORS_LIST + \
                      ['kdjk', 'open_2_sma', 'boll', 'close_10.0_le_5_c', 'wr_10', 'dma', 'trix']
for indicator in tech_indicator_list:
    _ = df[indicator]

# Add a new col: daily_return
df = df.copy()
df['daily_return'] = df.close.pct_change(1)
df = df.dropna()

# Split train/eval dataset
split_at = int(len(df) * 0.8)
df_val = df[split_at:]
df_val.reset_index(inplace=True)


# Reload model from disk
model = A2C.load('./trained_models/A2C_20201224-11h56.zip')

start = time.time()
val_env = DummyVecEnv([lambda: SinglePairEnv(df_val, 10000, 0.002, 1e-4, 18, tech_indicator_list, 150)])
obs = val_env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = val_env.step(action)
end = time.time()
print('Evaluation time: ', (end - start) / 60, ' minutes')
