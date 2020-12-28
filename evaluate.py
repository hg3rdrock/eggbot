import os
import time

import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stockstats import StockDataFrame as Sdf

from env import SinglePairEnv

# Read data from local csv file
data = pd.read_csv(os.path.join("data", "BTCUSDT-1d-data.csv"))
data.drop(columns=['close_time', 'quote_av', 'tb_base_av', 'tb_quote_av', 'ignore'], inplace=True)
data.rename(columns={'timestamp': 'date'}, inplace=True)

# Add tech indicator columns
df = Sdf.retype(data)
tech_indicator_list = ['macd', 'rsi_30', 'cci_30', 'dx_30', 'kdjk', 'open_2_sma', 'boll', 'close_10.0_le_5_c', 'wr_10',
                       'dma', 'trix']
for indicator in tech_indicator_list:
    _ = df[indicator]

# Add a new col: daily_return
df = df.copy()
df['daily_return'] = df.close.pct_change(1)
df = df.dropna()

# Split train/eval dataset
# split_at = int(len(df) * 0.8)
# df_val = df[split_at:]
# df_val = df[-365:-265]
df_val = df[-365:]
df_val.reset_index(inplace=True)

# Reload model from disk
# model = A2C.load('./trained_models/A2C_20201224-11h56.zip')
# model = A2C.load('./trained_models/A2C_20201224-16h47.zip')
# model = A2C.load('./trained_models/A2C_20201224-17h20.zip')
# model = A2C.load('./trained_models/A2C_20201224-20h24.zip')
# model = A2C.load('./trained_models/A2C_20201226-21h31.zip')
model = PPO.load('./trained_models/PPO_20201225-22h08.zip')

# model = DDPG.load('./trained_models/DDPG_20201224-16h16.zip')

def make_env(ds):
    return SinglePairEnv(ds, 10000, 0.002, 0.0001, 18, tech_indicator_list, 150)

def eval_model(model, ds):
    val_env = DummyVecEnv([lambda: make_env(ds)])
    obs = val_env.reset()
    done = False
    n_retrain_steps = 30
    n_step = 0

    start = time.time()

    while not done:
        action, _ = model.predict(obs)
        # action = val_env.action_space.sample()
        obs, reward, done, _ = val_env.step(action)
        n_step += 1
        # if n_step % n_retrain_steps == 0:
        #     tmp_ds = ds[n_step-30:n_step]
        #     model.set_env(DummyVecEnv([lambda: make_env(tmp_ds)]))
        #     model.learn(1200)
        #     print('model retrained')

    end = time.time()
    print('Evaluation time: ', (end - start) / 60, ' minutes')

    df_assets = val_env.env_method('save_asset_memory')
    # df_assets[0].plot(x='date', y='account_value')
    # backtestStats(df_assets[0], ds)
    buys, sells = val_env.env_method('saved_trades')[0]
    plt.figure(figsize=(16, 8))
    plt.plot(ds['close'])
    plt.plot(buys[:, 0], buys[:, 1], 'go')
    if len(sells) > 0:
        plt.plot(sells[:, 0], sells[:, 1], 'ro')
    plt.show()


# Evaluate model month by month
# total_val_days = len(df_val)
# start_day = 0
# while start_day <= total_val_days - 30:
#     print(f"{df_val.iloc[start_day]['date']} ====================")
#     dfv = df_val[start_day:start_day+30]
#     eval_model(model, dfv)
#     start_day += 30

# Evaluate model on the whole validation dataset
eval_model(model, df_val)
