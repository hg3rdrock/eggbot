import os, sys
import pandas as pd
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines3.common.callbacks import EvalCallback

from agent import DRLAgent
from env3 import CryptoPortfolioEnv


def read_data(csv_file):
    data = pd.read_csv(os.path.join("data", csv_file))

    split_at = int(len(data) * 0.8)
    train_data = data[:split_at]
    test_data = data[split_at:]
    return train_data, test_data


df_train1, df_val1 = read_data("Huobi_BTCUSDT_1h.csv")
df_train2, df_val2 = read_data("Huobi_BTC3S_1h.csv")

train_env = DummyVecEnv([lambda: CryptoPortfolioEnv(df_train1, df_train2)])
train_env = VecCheckNan(train_env, raise_exception=True)

agent = DRLAgent(env=train_env)

import datetime

now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
a2c_params_tuning = {'n_steps': 6,
                     'ent_coef': 0.005,
                     'learning_rate': 0.0007,
                     'verbose': 0,
                     'timesteps': 300000}

model = A2C('MlpPolicy', train_env,
            n_steps=a2c_params_tuning['n_steps'],
            ent_coef=a2c_params_tuning['ent_coef'],
            learning_rate=a2c_params_tuning['learning_rate'],
            verbose=a2c_params_tuning['verbose'],
            tensorboard_log=f"tensorboard_log/CryptoA2C_{now}"
            )
model.learn(100000)
model.save(f"trained_models/CryptoPfoA2C_{now}")
sys.exit("model trained")

# model_a2c = agent.train_A2C(model_name="CryptoA2C_{}".format(now), model_params=a2c_params_tuning)

# Validate the model
model = A2C.load('./trained_models/CryptoPfoA2C_20201230-21h24.zip')

val_env = DummyVecEnv([lambda: CryptoPortfolioEnv(df_val1, df_val2)])
print(f"buy_hold: {df_val1.iloc[30]['open']} ==> {df_val1.iloc[-1]['close']}")
obs = val_env.reset()
done = False

while not done:
    action, _ = model.predict(obs)
    _, _, done, _ = val_env.step(action)

# Plot trades

import matplotlib.pyplot as plt

buys, sells = val_env.env_method('saved_trades')[0]
plt.figure(figsize=(16, 8))
plt.plot(df_val1['close'])
if len(buys) > 0:
    plt.plot(buys[:, 0], buys[:, 1], 'go')
    print("buys =====")
    print(buys)
if len(sells) > 0:
    plt.plot(sells[:, 0], sells[:, 1], 'ro')
    print("sells =====")
    print(sells)
plt.show()
