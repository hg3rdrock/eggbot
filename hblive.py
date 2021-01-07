import ccxt
import numpy as np

exc = ccxt.huobipro({
    'apiKey': 'mk0lklo0de-bab860fb-e28d3608-eab36',
    'secret': '3448139b-848dc9aa-9424d4e1-ccfa8',
    'options': {
        'createMarketBuyOrderRequiresPrice': False
    }
})

# order = exc.create_market_sell_order('BTC3S/USDT', 200.00002345)
# order = exc.create_market_buy_order('BTC3S/USDT', 5.000235)
# print(order)

h1 = exc.fetchOHLCV('BTC/USDT', '1h', limit=30)
h2 = exc.fetchOHLCV('BTC3S/USDT', '1h', limit=30)
price1 = h1[-1][-2]
price2 = h2[-1][-2]
h1, h2 = np.array(h1), np.array(h2)
obs = np.stack((h1[:, -5:], h2[:, -5:]))

print(f'btc: {price1}, btc3s: {price2}')