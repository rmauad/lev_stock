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
quant_dlev = 10
quant_intan = 3
quant_lev = 5
quant_lev_vol = 0
# intan_strat = 0 # CAREFUL WITH THE GRAPH LEGEND. 1 for spltting between high and low intangible/assets ratio, 0 for splliting according to leverage level   
double_strat = 0 # 0 for single strategy (either intangibles or leverage level), 1 for double strategy.
holding_period = 1 # Number of months to hold the portfolio
window = 60 # Rolling Sharpe ratio window
window_vol = 12 # Rolling leverage volatility window
value_weight = 0 # 1 for value-weighted returns, 0 for equal-weighted returns
#################################################

df = dc.load_fm(redo = False)

df_reset = df.reset_index()

rf = df_reset[['year_month', 'rf']]
# rf = pd.read_csv('data/csv/.csv')

df = df.reset_index()
df = (df
      .rename(columns={'GVKEY': 'gvkey', 'RET': 'ret'})
      .drop(columns=['RET_lead1'])
)

#############################################################
# Buy stocks with low leverage change (quintile 1)
# and sell stocks with high leverage change (quintile 5)
# Do this for stocks with high intangible/assets ratio
# for those with low intangible/assets ratio and compare
# the results. Rebalance every month and hold for one month
##############################################################

# df.shape
# df_quant[['gvkey', 'year_month', 'd_debt_at_5', 'intan_at_3', 'debt_at_4']].tail(50)
df_quant = ist.create_quantiles(df, quant_dlev, quant_intan, quant_lev, quant_lev_vol, window_vol)
# df_lev_vol = ist.calc_lev_vol(df_port_ret, window_vol)

# df_port = ist.create_portfolios(df_quant, quant_dlev, quant_intan, quant_lev, quant_lev_vol, all_stocks, intan_strat, double_strat)
df_port = ist.create_portfolios(df_quant, quant_dlev, quant_intan, quant_lev, double_strat)

df_port_ret = ist.weighted_returns(df_port, holding_period, value_weight)

df_port_ret_agg = ist.agg_weighted_returns(df_port_ret, holding_period)
df_port_ret_agg_avr =  ist.avr_port_holding_period(df_port_ret_agg, holding_period)
df_strat_ret = ist.calc_avr_portfolio(df_port_ret_agg_avr)

###############################
# Graph the cumulative results
# and get the Sharpe ratio
###############################

# df_sp500, fig = ist.plot_returns_mkt(df_strat_ret, sp500, all_stocks)
df_strat, fig = ist.plot_returns(df_strat_ret, double_strat)

plt.savefig(figfolder + 'cum_ret_10_perc.pdf', format='pdf',  bbox_inches='tight')

###########################
# DON'T RUN THIS BLOCK YET
###########################

df_sharpe = ist.merge_df_rf(df_strat, rf)
sharpe_ratio, excess_returns, dates = ist.calculate_annualized_sharpe_ratio(df_sharpe)
rolling_sharpe_ratio = ist.rolling_sharpe_ratio(excess_returns, window, dates)
plot_sharpe = ist.plot_sharpe(rolling_sharpe_ratio, double_strat)


plt.savefig(figfolder + 'sr_ret_10_perc.pdf', format='pdf', bbox_inches='tight')
print("Finished")
# save = df_strat_ret.to_feather('data/feather/df_strat_ret.feather')

###########################################
# Calculate the recent leverage volatility
###########################################
