import numpy as np
from gym import spaces, Env
from sklearn.preprocessing import MinMaxScaler


class CryptoEnv(Env):

    def __init__(self, n_lookback, n_features, df):
        super(CryptoEnv, self).__init__()

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(n_lookback, n_features))
        self.df = df
        self.obs_df = df.copy()
        normalizer = MinMaxScaler()
        self.obs_df[['open', 'high', 'low', 'close', 'volume']] = normalizer.fit_transform(
            self.obs_df[['open', 'high', 'low', 'close', 'volume']])
        self.n_lookback = n_lookback
        self.n_features = n_features
        self.buy_actions = []
        self.sell_actions = []

    def reset(self):
        self.usdt_amt = 10000
        self.coin_amt = 0
        self.n_trades = 0
        self.ts = self.n_lookback
        obs = self._make_obs(self.ts)

        return obs

    def step(self, action):
        begin_balance = self._calc_balance()

        if action == 1:
            self._buy()
        elif action == 2:
            self._sell()

        self.ts += 1

        done = self.ts >= len(self.df) - 1
        obs = self._make_obs(self.ts)
        end_balance = self._calc_balance()
        reward = end_balance - begin_balance

        if done:
            print("initial balance: 10000")
            print(f"end balance: {end_balance}")
            print(f"total trades: {self.n_trades}")

        return obs, reward, done, {}

    def _make_obs(self, ts):
        obs = self.obs_df.iloc[ts - self.n_lookback:ts].values[:, -self.n_features:]
        return obs

    def _calc_balance(self):
        price = self.df.iloc[self.ts]['close']
        return self.coin_amt * price + self.usdt_amt

    def _buy(self):
        if self.usdt_amt > 0:
            price = self.df.iloc[self.ts]['close']
            self.coin_amt += self.usdt_amt * (1 - 0.002) / price
            self.usdt_amt = 0
            if self.n_trades == 0:
                self.buy_actions = []
            self.n_trades += 1
            self.buy_actions.append((self.ts, price))

    def _sell(self):
        if self.coin_amt > 0:
            price = self.df.iloc[self.ts]['close']
            self.usdt_amt += self.coin_amt * (1 - 0.002) * price
            self.coin_amt = 0
            if self.n_trades == 0:
                self.sell_actions = []
            self.n_trades += 1
            self.sell_actions.append((self.ts, price))

    def saved_trades(self):
        return np.array(self.buy_actions), np.array(self.sell_actions)
