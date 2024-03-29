import csv
import logging
import time

import ccxt
import numpy as np
from gym import spaces, Env


class HuobiLiveEnv(Env):

    def __init__(self, init_amt1, init_amt2, simulate_mode=True, max_steps=100):
        super(HuobiLiveEnv, self).__init__()

        self.init_amt1 = init_amt1
        self.init_amt2 = init_amt2
        self.simulate_mode = simulate_mode
        self.max_steps = max_steps

        self.exc = ccxt.huobipro({
            'apiKey': 'new-api-key',
            'secret': 'new-api-secret',
            'options': {
                'createMarketBuyOrderRequiresPrice': False
            }
        })

        self.pair1 = 'BTC/USDT'
        self.pair2 = 'BTC3S/USDT'

        self.n_lookback = 30
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2, self.n_lookback, 5), dtype=np.float)

    def reset(self):
        self.ts = 0
        self.assets = np.array([self.init_amt1, self.init_amt2], dtype=np.float)
        self.n_realloc = 0

        obs = self._make_obs()
        self.init_balance = self._calc_balance()

        return obs

    def _make_obs(self):
        h1 = self.exc.fetchOHLCV(self.pair1, '1h', limit=self.n_lookback)
        h2 = self.exc.fetchOHLCV(self.pair2, '1h', limit=self.n_lookback)
        self.price1 = h1[-1][-2]
        self.price2 = h2[-1][-2]
        h1, h2 = np.array(h1), np.array(h2)
        return np.stack((h1[:, -5:], h2[:, -5:]))

    def step(self, action):
        begin_balance = self._calc_balance()
        self._record_balance(self.assets, self.price1, self.price2, begin_balance)

        self._alloc_assets(action[0])

        time.sleep(60 * 60)
        self.ts += 1

        done = self.ts >= self.max_steps
        obs = self._make_obs()
        end_balance = self._calc_balance()
        reward = end_balance - begin_balance

        if self.ts % 24 == 0:
            print(f"=======step {self.ts}=======")
            print(f"initial balance: {self.init_balance}")
            print(f"end balance: {end_balance}")
            print(f"end assets: {self.assets}")
            print(f"total realloc: {self.n_realloc}")

        return obs, reward, done, {}

    def _alloc_assets(self, action):
        total_usdt = np.dot(self.assets, np.array([self.price1, self.price2]))
        frac1 = self.assets[0] * self.price1 / total_usdt

        if action > frac1 + 0.01:
            usdt_amt = (action - frac1) * total_usdt
            if usdt_amt <= 5:
                logging.debug(f"realloc usdt amt too low, action:{action}, frac1:{frac1}")
                return

            logging.info(f"timestep: {self.ts}")
            logging.info(f"sell btc3s and buy btc for {usdt_amt} USDT")

            delta_pair1 = usdt_amt * (1 - 0.004) / self.price1
            delta_pair2 = min(usdt_amt / self.price2, self.assets[1])

            self.assets[0] += delta_pair1
            self.assets[1] -= delta_pair2
            self.n_realloc += 1

            if not self.simulate_mode:
                ok = self._sell(self.pair2, delta_pair2)
                if ok:
                    time.sleep(2)
                    self._buy(self.pair1, usdt_amt * (1 - 0.002))

        elif action < frac1 - 0.01:
            usdt_amt = (frac1 - action) * total_usdt
            if usdt_amt <= 5:
                logging.error(f"realloc usdt amt too low, action:{action}, frac1:{frac1}")
                return
            logging.info(f"timestep: {self.ts}")
            logging.info(f"sell btc and buy btc3s for {usdt_amt} USDT")

            delta_pair1 = min(usdt_amt / self.price1, self.assets[0])
            delta_pair2 = usdt_amt * (1 - 0.004) / self.price2
            self.assets[0] -= delta_pair1
            self.assets[1] += delta_pair2
            self.n_realloc += 1

            if not self.simulate_mode:
                ok = self._sell(self.pair1, delta_pair1)
                if ok:
                    time.sleep(2)
                    self._buy(self.pair2, usdt_amt * (1 - 0.002))

    def _calc_balance(self):
        return np.dot(self.assets, np.array([self.price1, self.price2]))

    def _buy(self, symbol, amount):
        order = self.exc.create_market_buy_order(symbol, int(amount))
        return order['info']['status'] == 'ok'

    def _sell(self, symbol, amount):
        order = self.exc.create_market_sell_order(symbol, amount)
        return order['info']['status'] == 'ok'

    def _record_balance(self, assets, price1, price2, total_balance):
        with open('balance.csv', 'a+', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([self.ts, assets[0], assets[1], price1, price2, total_balance])

    def saved_trades(self):
        pass
