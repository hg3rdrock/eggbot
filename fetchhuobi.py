import os
import time

import ccxt
import numpy as np
import pandas as pd

exchange = ccxt.huobipro()


def fetch_save(symbol, csv_file):
    path_to_csv = os.path.join(os.curdir, 'data', csv_file)

    # from_ts = exchange.parse8601('2020-09-01T00:00:00')

    # data = exchange.fetchOHLCV(symbol, '1h', limit=2000)
    # df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'], dtype=np.float)
    # df.to_csv(path_to_csv)

    data = pd.read_csv(path_to_csv)
    last_ts = data.iloc[-1]['timestamp']
    while True:
        time.sleep(1)
        klines = exchange.fetchOHLCV(symbol, '1h', since=(last_ts + 3600 * 1000))
        if len(klines) == 0:
            break
        else:
            last_ts = klines[-1][0]
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'], dtype=np.float)
            df = df.set_index('timestamp')
            df.to_csv(path_to_csv, mode='a', header=False)


def clean_csv(csv_file):
    df = pd.read_csv(csv_file)
    df = df.drop(columns=['Unnamed: 0'])
    df = df.set_index('timestamp')
    df.to_csv(csv_file)


# clean_csv('./data/Huobi_BTCUSDT_1h.csv')
# clean_csv('./data/Huobi_BTC3S_1h.csv')

fetch_save('BTC/USDT', 'Huobi_BTCUSDT_1h.csv')
fetch_save('BTC3S/USDT', 'Huobi_BTC3S_1h.csv')
