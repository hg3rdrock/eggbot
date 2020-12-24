import random

import gym
import numpy as np
import pandas as pd
from gym import spaces
from sklearn.preprocessing import MinMaxScaler

MAX_STEPS = 10000
TRANSACTION_COST = 0.002
MAX_PRICE = 100000


class SingleCoinEnv(gym.Env):
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
        obs = [self.balance / self.max_balance, self.coin_held / 50]
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


class SinglePairEnv(gym.Env):

    def __init__(self,
                 df,
                 initial_amount,
                 transaction_cost_pct,
                 reward_scaling,
                 state_space,
                 tech_indicator_list,
                 turbulence_threshold,
                 ts=0):
        self.df = df
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.tech_indicator_list = tech_indicator_list
        self.turbulence_threshold = turbulence_threshold
        self.ts = ts

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(state_space,))
        self.data = self.df.iloc[self.ts]
        self.terminal = False
        self.state = [self.initial_amount] + [self.data.close] + [0] + \
                     sum([[self.data[ti]] for ti in tech_indicator_list], []) + \
                     [self.data.open, self.data.high, self.data.low, self.data.daily_return]

        self.reward = 0
        self.cost = 0
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self.data.date]
        self.close_price_memory = [self.data.close]
        self.trades = 0

    def reset(self):
        self.ts = 0
        self.data = self.df.iloc[0]
        self.terminal = False
        self.state = [self.initial_amount] + [self.data.close] + [0] + \
                     sum([[self.data[ti]] for ti in self.tech_indicator_list], []) + \
                     [self.data.open, self.data.high, self.data.low, self.data.daily_return]

        self.reward = 0
        self.cost = 0
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self.data.date]
        self.trades = 0

        return self.state

    def step(self, actions):
        self.terminal = self.ts >= len(self.df) - 1

        if self.terminal:
            end_total_asset = self.state[0] + self.state[1] * self.state[2]
            print("begin_total_asset:{}".format(self.asset_memory[0]))
            print("end_total_asset:{}".format(end_total_asset))
            print(f"total reward: {end_total_asset - self.initial_amount}")
            print(f"total cost: {self.cost}")
            print(f"total trades: {self.trades}")

            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.columns = ['account_value']
            df_total_value['daily_return'] = df_total_value.pct_change(1)
            if df_total_value['daily_return'].std() != 0:
                sharpe = ((24 * 7) ** 0.5) * df_total_value['daily_return'].mean() / \
                         df_total_value['daily_return'].std()
                print(f"Sharpe: {sharpe}")
                print("=================================")
        else:
            begin_total_asset = self.state[0] + self.state[1] * self.state[2]

            action = actions[0]
            if action < -0.5:
                self._sell(2 * abs(action) - 1)
            elif action > 0.5:
                self._buy(2 * action - 1)

            self.ts += 1
            self.data = self.df.iloc[self.ts]
            self.state = [self.state[0]] + [self.data.close] + [self.state[2]] + \
                         sum([[self.data[ti]] for ti in self.tech_indicator_list], []) + \
                         [self.data.open, self.data.high, self.data.low, self.data.daily_return]

            end_total_asset = self.state[0] + self.state[1] * self.state[2]
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self.data.date)
            self.close_price_memory.append(self.data.close)
            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling

        return self.state, self.reward, self.terminal, {}

    def _sell(self, action):
        if self.state[2] > 0:
            transaction_cost = self.state[2] * action * self.transaction_cost_pct * self.state[1]
            self.cost += transaction_cost

            self.state[0] += self.state[2] * action * self.state[1] - transaction_cost
            self.state[2] -= self.state[2] * action

            self.trades += 1
        else:
            pass

    def _buy(self, action):
        if self.state[0] > 0:
            transaction_cost = self.state[0] * action * self.transaction_cost_pct
            self.cost += transaction_cost

            self.state[2] += (self.state[0] * action - transaction_cost) / self.state[1]
            self.state[0] -= self.state[0] * action

            self.trades += 1
        else:
            pass

    def render(self, mode='human'):
        pass
