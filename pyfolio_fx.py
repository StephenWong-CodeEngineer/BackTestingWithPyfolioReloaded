
import warnings

from numpy.f2py.crackfortran import endifs

warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt

from findatapy.market import Market, MarketDataGenerator, MarketDataRequest

import pyfolio as pf

try:
    FRED_API_KEY = os.environ['FRED_API_KEY']
except:
    pass

filepath2 = './data/'


if not os.path.exists(filepath2 + 'daily_fx_spot_data.csv'):

    md_request = MarketDataRequest(
        start_date='01 Jan 1989',  # Start date
        tickers=['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'NZDUSD', 'USDCHF', 'USDNOK', 'USDSEK'],   # What we want the ticker to look like once download
        vendor_tickers=['DEXUSEU', 'DEXJPUS', 'DEXUSUK', 'DEXUSAL', 'DEXCAUS', 'DEXUSNZ', 'DEXSZUS', 'DEXNOUS', 'DEXSDUS' ],  # The ticker used by the vendor
        fields=['close'],  # What fields we want (usually close, we can also define vendor fields)
        data_source='alfred',  # What is the data source?
        fred_api_key=FRED_API_KEY)  # Most data sources will require us to specify an API key/password

    market = Market()
    df_daily_fx = market.fetch_market(md_request)

    print(df_daily_fx[:10].to_string())
    print('\n')
    print(df_daily_fx[-10:].to_string())
    print('\n')

    df_daily_fx.to_csv(filepath2 + 'daily_fx_spot_data.csv')

else:

    df_daily_fx = pd.read_csv(filepath2 + 'daily_fx_spot_data.csv', index_col=0)

df_daily_fx.index = pd.to_datetime(df_daily_fx.index, utc=True)    # Converting to utc or Event Study will crash!

print('df_daily_fx:')
print(df_daily_fx[:10].to_string())
print(df_daily_fx[-10:].to_string())
print()

df_daily_fx_rets = df_daily_fx / df_daily_fx.shift(1) - 1.0

print('df_daily_fx_rets:')
print(df_daily_fx_rets[:10].to_string())
print(df_daily_fx_rets[-10:].to_string())
print()

num_currencies = df_daily_fx.columns.size
num_days_per_year = 252
tc = 2.0 / 100.0 / 100.0 / 2.0      # assume 2 bp

days_of_average = [20, 50, 100, 200, 300]

max_info_ratio = np.array([-99999.0 for i in range(num_currencies)])
best_num_days_of_mv_avg = np.array([0 for i in range(num_currencies)])


for ic, day_avg in enumerate(days_of_average):
    df_sma_fx = df_daily_fx.rolling(day_avg).mean()
    fx_signal = np.where(df_daily_fx > df_sma_fx, 1.0, -1.0)

    df_fx_signal = pd.DataFrame(index=df_daily_fx.index, data=fx_signal, columns=df_daily_fx.columns)

    df_strat_fx_ret = df_fx_signal.shift(1) * df_daily_fx_rets \
                       - abs(df_fx_signal - df_fx_signal.shift(1)) * tc

    print('df_strat_fx_ret:')
    print(df_strat_fx_ret[:10].to_string())
    print(df_fx_signal[:10].to_string())
    print()
    print(df_strat_fx_ret[-10:].to_string())
    print(df_fx_signal[-10:].to_string())

    ann_ret = df_strat_fx_ret.mean(axis=0) * num_days_per_year
    ann_vol = df_strat_fx_ret.std(axis=0) * math.sqrt(num_days_per_year)
    info_ratio = ann_ret / ann_vol
    print('info_ratio:\n', info_ratio)
    print()
    filter = max_info_ratio < info_ratio
    max_info_ratio[filter] = info_ratio[filter]
    best_num_days_of_mv_avg[filter] = day_avg

print('Best info ratio ', max_info_ratio)
print('Best num of days of moving average to use ', best_num_days_of_mv_avg)
print()


df_fx_signals = pd.DataFrame(index=df_daily_fx.index)
df_strat_fx_rets = pd.DataFrame(index=df_daily_fx.index)

fx_signals = np.array([] for i in range(num_currencies))
for col, days in zip(df_daily_fx.columns, best_num_days_of_mv_avg):
    df_sma_fx = df_daily_fx[col].rolling(window=days, min_periods=1).mean()
    df_fx_signals[col] = np.where(df_daily_fx[col] > df_sma_fx, 1.0, -1.0)

    df_strat_fx_rets[col] = df_fx_signals[col].shift(1) * df_daily_fx_rets[col] \
                                    - abs(df_fx_signals[col] - df_fx_signals[col].shift(1)) * tc


print('df_strat_fx_rets and df_fx_signals:')
print(df_strat_fx_rets[:10].to_string())
print(df_fx_signals[:10].to_string())
print()
print(df_strat_fx_rets[-10:].to_string())
print(df_fx_signals[-10:].to_string())

df_strat_fx_cum_rets = 100.0 * (1.0 + df_strat_fx_rets).cumprod()

df_strat_fx_cum_rets.plot()
plt.title('Cumulative Returns on different no. of days of moving average per currency')
plt.show()


print('Pyfolio')

for bd, c in zip(best_num_days_of_mv_avg, df_strat_fx_rets.columns):
    series = pd.Series(index=df_strat_fx_rets.index, data=df_strat_fx_rets[c])

    print('**********************************************************************************')
    print(c + ' ' + str(bd) + ' days average:')
    pf.create_returns_tear_sheet(series, live_start_date='2024-01-01')
    plt.show()


