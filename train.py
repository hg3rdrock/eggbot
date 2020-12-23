import os

import pandas as pd
import numpy as np
from finrl.config import config
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stockstats import StockDataFrame as Sdf

from env import SinglePairEnv


# Read data from local csv file
data = pd.read_csv(os.path.join("data", "BTCUSDT-1h-data.csv"))
data = data.dropna()

# Add tech indicator columns
df = Sdf.retype(data)
tech_indicator_list = config.TECHNICAL_INDICATORS_LIST + ['kdjk', 'open_2_sma', 'boll', 'close_10.0_le_5_c', 'wr_10',
                                                          'dma', 'trix']
for indicator in tech_indicator_list:
    _ = df[indicator]

# Add a new col: daily_return
df = df.copy()
df['daily_return'] = df.close.pct_change(1)
df = df.fillna(method='bfill').fillna(method="ffill")

# Split train/eval dataset
split_at = int(len(df) * 0.8)
df_train = df[:split_at]
df_val = df[split_at:]
print(df_train.head())

feature_list = ['open', 'high', 'low', 'close', 'volume'] + tech_indicator_list
print(feature_list)
train_env = DummyVecEnv([lambda: SinglePairEnv(df_train, 10000, feature_list)])

model = PPO('MlpPolicy', train_env, verbose=1)
# action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros((1, 2)), sigma=float(0.5) * np.ones((1, 2)))
# model = DDPG('MlpPolicy', train_env)
model.learn(total_timesteps=300000)

val_env = DummyVecEnv([lambda: SinglePairEnv(df_val, 10000, feature_list)])
obs = val_env.reset()
for i in range(len(df_val)):
    action, _ = model.predict(obs)
    obs, rewards, done, _ = val_env.step(action)
    val_env.render()
