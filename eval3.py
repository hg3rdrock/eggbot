import os
import sys

import ccxt
import numpy as np
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv

from env3 import CryptoPortfolioEnv


def read_data(csv_file):
    data = pd.read_csv(os.path.join("data", csv_file))

    split_at = int(len(data) * 0.8)
    train_data = data[:split_at]
    test_data = data[split_at:]
    return train_data, test_data


_, df_val1 = read_data("Huobi_BTCUSDT_1h.csv")
_, df_val2 = read_data("Huobi_BTC3S_1h.csv")


# Read live data
def read_live_data(limit=120):
    exc = ccxt.huobipro()
    data1 = exc.fetchOHLCV('BTC/USDT', '1h', limit=limit)
    df1 = pd.DataFrame(data1, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'], dtype=np.float32)
    data2 = exc.fetchOHLCV('BTC3S/USDT', '1h', limit=limit)
    df2 = pd.DataFrame(data2, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'], dtype=np.float32)
    return df1, df2


df_val1, df_val2 = read_live_data(24 * 7 + 30)
print(f"btc price {df_val1.iloc[0]['close']} ==> {df_val1.iloc[-1]['close']}")

# Validate the model
model = A2C.load('./trained_models/CryptoPfoA2C_20210119-16h04.zip')
# model = A2C.load('./trained_models/CryptoPfoPPO_20210107-21h41.zip')

val_env = DummyVecEnv([lambda: CryptoPortfolioEnv(df_val1, df_val2, training=False)])
obs = val_env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=False)
    obs, _, done, _ = val_env.step(action)

print(f"buy_hold: {df_val1.iloc[30]['open']} ==> {df_val1.iloc[-1]['close']}")

sys.exit("model evaluation done")
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
