import os

import ccxt
import numpy as np
import pandas as pd

exchange = ccxt.huobipro()


def fetch_save(symbol, csv_file):
    data = exchange.fetchOHLCV(symbol, '1h')
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'], dtype=np.float)
    df.to_csv(os.path.join(os.curdir, 'data', csv_file))


fetch_save('BTC/USDT', 'Huobi_BTCUSDT_1h.csv')
fetch_save('BTC3S/USDT', 'Huobi_BTC3S_1h.csv')
