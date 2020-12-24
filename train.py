import os
import time

import pandas as pd
from finrl.config import config
from finrl.model.models import DRLAgent
from stable_baselines3.common.vec_env import DummyVecEnv
from stockstats import StockDataFrame as Sdf

from env import SinglePairEnv

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

agent = DRLAgent(env=train_env)

# Train the agent
import datetime

now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
a2c_params_tuning = {'n_steps': 5,
                     'ent_coef': 0.005,
                     'learning_rate': 0.0007,
                     'verbose': 0,
                     'timesteps': 200000}
model_a2c = agent.train_A2C(model_name="A2C_{}".format(now), model_params=a2c_params_tuning)

# Validate the model

start = time.time()
# dfv = df_val[-60:]
val_env = DummyVecEnv([lambda: SinglePairEnv(df_val, 10000, 0.002, 1e-4, state_space, tech_indicator_list, 150)])
obs = val_env.reset()
done = False
while not done:
    action, _ = model_a2c.predict(obs)
    obs, reward, done, _ = val_env.step(action)
end = time.time()
print('Evaluation time: ', (end - start) / 60, ' minutes')
