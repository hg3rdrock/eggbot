import os

import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stockstats import StockDataFrame as Sdf

from agent import DRLAgent
from env import SinglePairEnv

# Read data from local csv file
# data = pd.read_csv(os.path.join("data", "BTCUSDT-1d-data.csv"))
# data.drop(columns=['close_time', 'quote_av', 'tb_base_av', 'tb_quote_av', 'ignore'], inplace=True)
# data.rename(columns={'timestamp': 'date'}, inplace=True)
data = pd.read_csv(os.path.join("data", "BTC-USD.csv"))
print(f"training days: {len(data)}")
data.columns = data.columns.str.lower()

# Add tech indicator columns
df = Sdf.retype(data)
tech_indicator_list = ['macd', 'rsi_30', 'cci_30', 'dx_30', 'kdjk', 'open_2_sma', 'boll', 'close_10.0_le_5_c', 'wr_10',
                       'dma', 'trix']
for indicator in tech_indicator_list:
    _ = df[indicator]

# Add a new col: daily_return
df = df.copy()
df['daily_return'] = df.close.pct_change(1)
# df = df.fillna(method='bfill').fillna(method="ffill")
df = df.dropna()

# Split train/eval dataset
split_at = int(len(df) * 0.8)
df_train = df[:split_at]
df_val = df[split_at:]
df_train.reset_index(inplace=True)
df_val.reset_index(inplace=True)
print(df_train.head())

# feature_list = ['open', 'high', 'low', 'close', 'volume'] + tech_indicator_list
# print(feature_list)
# train_env = DummyVecEnv([lambda: SinglePairEnv(df_train, 10000, feature_list)])
# model = PPO('MlpPolicy', train_env, verbose=1)
# action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros((1, 2)), sigma=float(0.5) * np.ones((1, 2)))
# model = DDPG('MlpPolicy', train_env)
# model.learn(total_timesteps=300000)


# Create env and agent
state_space = 7 + len(tech_indicator_list)
train_env = DummyVecEnv([lambda: SinglePairEnv(df_train, 10000, 0.002, 1e-4, state_space, tech_indicator_list, 150)])
train_env = VecCheckNan(train_env, raise_exception=True)

agent = DRLAgent(env=train_env)

# Train the agent
import datetime

# now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
# a2c_params_tuning = {'n_steps': 24,
#                      'ent_coef': 0.005,
#                      'learning_rate': 0.0007,
#                      'verbose': 0,
#                      'timesteps': 50000}
# model_a2c = agent.train_A2C(model_name="A2C_{}".format(now), model_params=a2c_params_tuning)

# now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
# ddpg_params_tuning = {
#     'batch_size': 128,
#     'buffer_size': 100000,
#     'learning_rate': 0.0003,
#     'verbose': 0,
#     'timesteps': 30000}
# model_ddpg = agent.train_DDPG(model_name="DDPG_{}".format(now), model_params=ddpg_params_tuning)

# now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
# td3_params_tuning = {
#     'batch_size': 100,
#     'buffer_size': 1000000,
#     'learning_rate': 0.001,
#     'verbose': 0,
#     'timesteps': 30000}
# model_td3 = agent.train_TD3(model_name="TD3_{}".format(now), model_params=td3_params_tuning)

# now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
# ppo_params_tuning = {'n_steps': 128,
#                      'batch_size': 64,
#                      'ent_coef': 0.005,
#                      'learning_rate': 0.025,
#                      'verbose': 0,
#                      'timesteps': 500000}
# model_ppo = agent.train_PPO(model_name="PPO_{}".format(now), model_params=ppo_params_tuning)

now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
sac_params_tuning = {
    'batch_size': 64,
    'buffer_size': 2000,
    'ent_coef': 'auto_0.1',
    'learning_rate': 0.0001,
    'learning_starts': 100,
    'timesteps': 10000,
    'verbose': 1}
model_sac = agent.train_SAC(model_name="SAC_{}".format(now), model_params=sac_params_tuning)
