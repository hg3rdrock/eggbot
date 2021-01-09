import os

import ccxt
import numpy as np
import pandas as pd

exchange = ccxt.huobipro()


def fetch_save(symbol, csv_file):
    path_to_csv = os.path.join(os.curdir, 'data', csv_file)

    # from_ts = exchange.parse8601('2020-09-01T00:00:00')

    data = exchange.fetchOHLCV(symbol, '1h', limit=2000)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'], dtype=np.float)
    df.to_csv(path_to_csv)

    # last_ts = data[-1][0]
    # while True:
    #     time.sleep(1)
    #     klines = exchange.fetchOHLCV(symbol, '1h', since=last_ts)
    #     last_ts = klines[-1][0]
    #     if len(klines) == 0:
    #         break
    #     else:
    #         df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'], dtype=np.float)
    #         df.to_csv(path_to_csv, mode='a', header=False)


fetch_save('BTC/USDT', 'Huobi_BTCUSDT_1h.csv')
fetch_save('BTC3S/USDT', 'Huobi_BTC3S_1h.csv')
