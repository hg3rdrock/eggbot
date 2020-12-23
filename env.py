import random
import numpy as np
import gym
from gym import spaces
from sklearn.preprocessing import MinMaxScaler


MAX_STEPS = 10000
TRANSACTION_COST = 0.002
MAX_PRICE = 100000

class SinglePairEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance, feature_list):
        super(SinglePairEnv, self).__init__()
        self.initial_balance = initial_balance
        self.feature_list = feature_list
        self.max_balance = initial_balance * 10
        self.df = df

        self.normalized_df = df.copy()
        normalizer = MinMaxScaler()
        self.normalized_df[feature_list] = normalizer.fit_transform(df[feature_list])

        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float)
        self.observation_space = spaces.Box(low=0, high=1, shape=(18,), dtype=np.float)
        pass

    def reset(self):
        self.ts = random.randint(0, len(self.df) - 24 * 7)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.coin_held = 0
        self.total_cost = 0

        return self._next_obs()

    def _next_obs(self):
        dp = self.normalized_df.iloc[self.ts]
        obs = [self.balance/self.max_balance, self.coin_held/50]
        for feature in self.feature_list:
            obs.append(dp[feature])

        return np.array(obs)

    def step(self, action):
        self._take_action(action)

        self.ts += 1
        if self.ts >= len(self.df) - 1:
            self.ts = 0

        delay_modifier = self.ts / MAX_STEPS
        reward = self.balance * delay_modifier
        done = self.net_worth <= 100
        obs = self._next_obs()

        return obs, reward, done, {}

    def _take_action(self, action):
        current_price = self.df.iloc[self.ts]["close"]
        action_type = action[0]
        amount_pct = action[1]

        if action_type < 1:
            # Buy coin
            self.coin_held = self.balance * amount_pct * (1 - TRANSACTION_COST) / current_price
            self.balance *= 1 - amount_pct
        elif action_type < 2:
            # Sell coin
            self.balance += self.coin_held * amount_pct * (1 - TRANSACTION_COST) * current_price
            self.coin_held *= 1 - amount_pct

        self.net_worth = self.balance + self.coin_held * current_price
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

    def render(self, mode='human'):
        profit = self.net_worth - self.initial_balance

        print(f"Step: {self.ts}")
        print(f"USDT: {self.balance:.2f}, BTC: {self.coin_held:.4f}")
        print(f"Net worth: {self.net_worth:.2f}")
        print(f"Profit: {profit:.2f}")
