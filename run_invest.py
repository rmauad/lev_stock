import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('code/lev_stock/invest_strat/')
import invest_strat as ist
import utils as ut
import datacleaning as dc
import os
# Silence warnings
import warnings
warnings.filterwarnings("ignore")

figfolder = ut.to_output('graph/')
# Defining variables for the investment strategy:

#################################################
# Define quantiles and holding period
print("Starting")
quant_dlev = 5
quant_intan = 3
quant_lev = 4
quant_lev_vol = 0
quant_kkr = 5
quant_pd = 5 # probability of default (Merton model)
quant_size = 2 # keep at 2
quant_bm = 3
# intan_strat = 0 # CAREFUL WITH THE GRAPH LEGEND. 1 for spltting between high and low intangible/assets ratio, 0 for splliting according to leverage level   
double_strat = 0 # 0 for single strategy (either intangibles or leverage level), 1 for double strategy.
# months_to_strat = 3 # Number of months after quarter-end date apply the strategy
holding_period = 1 # Number of months to hold the portfolio
window = 60 # Rolling Sharpe ratio window
window_vol = 12 # Rolling leverage volatility window
value_weight = 1 # 1 for value-weighted returns, 0 for equal-weighted returns
intan_measure = 'epk' # 'kkr' for KKR intangibles, 'epk' for Eisfeldt-Papanikolaou intangibles
subsample = 'all' # 'all' for all stocks, 'hint' for high IK/A, 'lint' for low IK/A, 'hpd' for high probability of default, 'lpd' for low probability of default
strat = 'lev' # 'lev' for leverage strategy, 'intan' for intangible strategy
#################################################

# dc.csv_to_pickle() # Run this line to convert the csv files to pickle files

df = dc.load_fm(redo = False)

# sp500 = pd.read_excel('../data/excel/sp500.xlsx')
# df = df.sample(n = 1000)

# kkr = pd.read_pickle('../data/pickle/kkr.pkl')
# kkr = (kkr
#       .assign(GVKEY = kkr['gvkey'].astype('Int64'))
#       )
# kkr = kkr[['GVKEY', 'year', 'KKR']]
# df = df.sort_values(by=['GVKEY', 'year_month'])
# col_intan_fill = ['KKR']

# # df_test = (pd.merge(df, kkr, how = 'left', on = ['GVKEY', 'year']))
# df[col_intan_fill] = df.groupby('GVKEY')[col_intan_fill].transform(lambda x: x.ffill(limit = 11))


# lb = df['RET'].quantile(0.05)
# up = df['RET'].quantile(0.95)
# df = df[(df['RET'] > lb) & (df['RET'] < up)]

# df = df.groupby('year_month').apply(lambda x: x.sample(1500)).reset_index(drop=True)

rf = df[['year_month', 'rf']]
# rf = pd.read_csv('data/csv/.csv')
df = (df
      .rename(columns={'GVKEY': 'gvkey', 'RET': 'ret'})
      .drop(columns=['RET_lead1'])
)
# df = df[df['ff_indust'] == 3]
# #############################################################
# # Buy stocks with low leverage change (quintile 1)
# # and sell stocks with high leverage change (quintile 5)
# # Do this for stocks with high intangible/assets ratio
# # for those with low intangible/assets ratio and compare
# # the results. Rebalance every month and hold for one month
# ##############################################################

# df.shape
# df_quant[['gvkey', 'year_month', 'd_debt_at_5', 'intan_at_3', 'debt_at_4']].tail(50)
# df_quant = ist.create_quantiles(df, quant_dlev, quant_intan, quant_lev, quant_kkr, quant_pd, quant_lev_vol, window_vol)


df_filtered = ist.filter_data(df)
df_quantiles = ist.create_quantiles(df_filtered, quant_dlev, quant_intan, quant_lev, quant_kkr, quant_pd, quant_size, quant_bm, quant_lev_vol, window_vol)

# std_strat = ist.calc_std_strat(df_quantiles, quant_size, quant_bm)


# df_subsample = (df_quantiles
#                 .assign(year = lambda x: x['year'].astype('Int64'))
#                 .query('year >= 2018')
#                 )

# df_subsample_test = ist.select_firms(df_subsample, quant_intan, quant_lev, quant_dlev)

# df_subsample_test.to_excel('data_sample.xlsx', sheet_name='data', index=False)

# df_lev_vol = ist.calc_lev_vol(df_port_ret, window_vol)

# df_port = ist.create_portfolios(df_quant, quant_dlev, quant_intan, quant_lev, quant_lev_vol, all_stocks, intan_strat, double_strat)
df_port = ist.create_portfolios(df_quantiles, quant_dlev, quant_intan, quant_lev, quant_kkr, quant_pd, double_strat, intan_measure, subsample)

df_port_ret = ist.weighted_returns(df_port, holding_period, value_weight)

df_port_ret_agg = ist.agg_weighted_returns(df_port_ret, holding_period)
df_port_ret_agg_avr =  ist.avr_port_holding_period(df_port_ret_agg, holding_period)
df_strat_ret = ist.calc_avr_portfolio(df_port_ret_agg_avr)

# df_strat_ret.to_excel('df_strat_ret.xlsx', index = False)

# ist.plot_returns_mkt(df_strat_ret, sp500, 1)
###############################
# Graph the cumulative results
# and get the Sharpe ratio
###############################

# df_sp500, fig = ist.plot_returns_mkt(df_strat_ret, sp500, all_stocks)

df_strat, fig = ist.plot_returns(df_strat_ret, double_strat, subsample, strat)

plt.savefig(figfolder + 'cumret.pdf', format='pdf',  bbox_inches='tight')

df_strat, fig = ist.plot_std(df_strat_ret, double_strat, subsample, strat)

plt.savefig(figfolder + 'std.pdf', format='pdf',  bbox_inches='tight')

df_sharpe = ist.merge_df_rf(df_strat, rf)
sharpe_ratio, excess_returns, dates = ist.calculate_annualized_sharpe_ratio(df_sharpe)
rolling_sharpe_ratio = ist.rolling_sharpe_ratio(excess_returns, window, dates)
plot_sharpe = ist.plot_sharpe(rolling_sharpe_ratio, double_strat, strat)


plt.savefig(figfolder + 'sr.pdf', format='pdf', bbox_inches='tight')
print("Finished")
# save = df_strat_ret.to_feather('data/feather/df_strat_ret.feather')

# ###########################
# # Plot HML and SMB returns
# ###########################


