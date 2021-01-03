import os
import pandas as pd
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines3.common.callbacks import EvalCallback

from agent import DRLAgent
from env2 import CryptoEnv


data = pd.read_csv(os.path.join("data", "BTCUSDT-4h-data.csv"))
data.drop(columns=['close_time', 'quote_av', 'tb_base_av', 'tb_quote_av', 'ignore'], inplace=True)
data.rename(columns={'timestamp': 'date'}, inplace=True)

split_at1 = int(len(data) * 0.6)
split_at2 = int(len(data) * 0.8)
df_train = data[:split_at1]
# df_val = data[split_at1:split_at2]
# df_val = data[split_at2:int(len(data) * 0.9)]
df_val = data[split_at2:]
df_val.reset_index(inplace=True)

train_env = DummyVecEnv([lambda :CryptoEnv(30, 5, df_train)])
train_env = VecCheckNan(train_env, raise_exception=True)

agent = DRLAgent(env=train_env)

import datetime

now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
a2c_params_tuning = {'n_steps': 36,
                     'ent_coef': 0.005,
                     'learning_rate': 0.0007,
                     'verbose': 0,
                     'timesteps': 300000}
val_env = DummyVecEnv([lambda :CryptoEnv(30, 5, df_val)])
val_env = VecCheckNan(val_env, raise_exception=True)
eval_cb = EvalCallback(val_env, best_model_save_path='./best_models/', eval_freq=len(df_train), deterministic=True, render=False)
model = A2C('MlpPolicy', train_env,
            n_steps=a2c_params_tuning['n_steps'],
            ent_coef=a2c_params_tuning['ent_coef'],
            learning_rate=a2c_params_tuning['learning_rate'],
            verbose=a2c_params_tuning['verbose'],
            tensorboard_log=f"tensorboard_log/CryptoA2C_{now}"
            )
# model.learn(300000)
# model.save(f"trained_models/CryptoA2C_{now}")

# model_a2c = agent.train_A2C(model_name="CryptoA2C_{}".format(now), model_params=a2c_params_tuning)

# Validate the model
model = A2C.load('./trained_models/CryptoA2C_20201230-21h24.zip')

val_env = DummyVecEnv([lambda :CryptoEnv(30, 5, df_val)])
print(f"buy_hold: {df_val.iloc[30]['open']} ==> {df_val.iloc[-1]['close']}")
obs = val_env.reset()
done = False

while not done:
    action, _ = model.predict(obs)
    _, _, done, _ = val_env.step(action)

# Plot trades

import matplotlib.pyplot as plt

buys, sells = val_env.env_method('saved_trades')[0]
plt.figure(figsize=(16, 8))
plt.plot(df_val['close'])
if len(buys) > 0:
    plt.plot(buys[:, 0], buys[:, 1], 'go')
    print(buys)
if len(sells) > 0:
    plt.plot(sells[:, 0], sells[:, 1], 'ro')
plt.show()
