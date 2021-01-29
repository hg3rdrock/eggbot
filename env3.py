import numpy as np
from gym import spaces, Env


class CryptoPortfolioEnv(Env):

    def __init__(self, df1, df2, training=True):
        super(CryptoPortfolioEnv, self).__init__()

        self.df1 = df1
        self.df2 = df2
        self.training = training
        self.obs_df1 = df1.copy()
        self.obs_df2 = df2.copy()
        self.n_lookback = 30
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2, self.n_lookback, 5), dtype=np.float)

    def reset(self):
        self.ts = self.n_lookback
        self.assets = np.array([1.0, 0.0], dtype=np.float)
        self.n_realloc = 0
        self.init_balance = self._calc_balance()

        obs = self._make_obs(self.ts)
        return obs

    def _make_obs(self, ts):
        h1 = self.obs_df1.iloc[ts - self.n_lookback:ts].values[:, -5:]
        h2 = self.obs_df2.iloc[ts - self.n_lookback:ts].values[:, -5:]
        return np.stack((h1, h2))

    def step(self, action):
        begin_balance = self._calc_balance()

        self._alloc_assets(action[0])

        self.ts += 1
        done = self.ts >= len(self.df2) - 1
        obs = self._make_obs(self.ts)
        end_balance = self._calc_balance()
        reward = end_balance - begin_balance

        if done:
            reward *= (self.assets[0] ** 2)
            print(f"initial balance: {self.init_balance}")
            print(f"end balance: {end_balance}")
            print(f"end assets: {self.assets}")
            print(f"total realloc: {self.n_realloc}")

        return obs, reward, done, {}

    def _alloc_assets(self, action):
        price1 = self.df1.iloc[self.ts]['close']
        price2 = self.df2.iloc[self.ts]['close']
        total_usdt = np.dot(self.assets, np.array([price1, price2]))
        frac1 = self.assets[0] * price1 / total_usdt

        if action > frac1 + 0.01:
            usdt_amt = (action - frac1) * total_usdt
            discount = 1.0 if self.training else (1 - 0.004)
            self.assets[0] += usdt_amt * discount / price1
            self.assets[1] -= min(usdt_amt / price2, self.assets[1])
            self.n_realloc += 1
            if not self.training:
                print(f"timestep: {self.ts}")
                print(f"sell btc3s and buy btc for {usdt_amt} USDT")
        elif action < frac1 - 0.01:
            usdt_amt = (frac1 - action) * total_usdt
            discount = 1.0 if self.training else (1 - 0.004)
            self.assets[0] -= min(usdt_amt / price1, self.assets[0])
            self.assets[1] += usdt_amt * discount / price2
            self.n_realloc += 1
            if not self.training:
                print(f"timestep: {self.ts}")
                print(f"sell btc and buy btc3s for {usdt_amt} USDT")

    def _calc_balance(self):
        price1 = self.df1.iloc[self.ts]['close']
        price2 = self.df2.iloc[self.ts]['close']

        return np.dot(self.assets, np.array([price1, price2]))


    def saved_trades(self):
        pass
