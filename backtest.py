import pandas as pd
import pyfolio as pf


def backtestStats(account_value, coin_value):
    df = account_value.copy()
    df['daily_return'] = df.account_value.pct_change(1)
    base_df = coin_value.copy()
    base_df['daily_return'] = base_df.close.pct_change(1)
    ds = backtest_strat(df)
    ds_base = backtest_strat(base_df)
    # pf.create_returns_tear_sheet(ds)

    with pf.plotting.plotting_context(font_scale=1.1):
        pf.create_full_tear_sheet(returns=ds, benchmark_rets=ds_base, set_context=False)

def backtest_strat(df):
    strategy_ret = df.copy()
    strategy_ret['date'] = pd.to_datetime(strategy_ret['date'], format="%Y-%m-%d")
    strategy_ret.set_index('date', drop=False, inplace=True)
    strategy_ret.index = strategy_ret.index.tz_localize('UTC')
    del strategy_ret['date']
    ts = pd.Series(strategy_ret['daily_return'].values, index=strategy_ret.index)
    return ts
