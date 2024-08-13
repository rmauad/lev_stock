import pandas as pd
# from tabulate import tabulate
import sys
sys.path.append('code/lev_stock/portfolio/')
sys.path.append('code/lev_stock/invest_strat/')
import portfolio as pf
import datacleaning as dc
import invest_strat as ist
import matplotlib.pyplot as plt
import utils as ut
# import portfolio_functions as pf

figfolder = ut.to_output('graph/')

# df = pd.read_feather('data/feather/df_fm.feather') # from prep_fm.py
df = dc.load_fm(redo = False)

df.set_index(['GVKEY', 'year_month'], inplace=True)
# df = df.reset_index()
# df = (df
#       .rename(columns={'GVKEY': 'gvkey'}))
#     #   .drop(columns=['RET_lead1'])
# df.set_index(['gvkey', 'year_month'], inplace=True)

#######################################
quant_dlev = 5
quant_intan = 5
quant_lev = 5
quant_kkr = 5
quant_pd = 5 # probability of default (Merton model)
intan_subsample = quant_intan # 1 for low intangibles, quant_intan for high.
quant_lev_vol = 0
window_vol = 12
#######################################

# Print paper tables (summary statistics, Fama-MacBeth regressions, etc.)
pf.create_sum_stats(df)
fig = ist.plot_intan_pd(df)

plt.savefig(figfolder + 'intan_pd.pdf', format='pdf', bbox_inches='tight')

df_quantiles = ist.create_quantiles(df, quant_dlev, quant_intan, quant_lev, quant_kkr, quant_pd, quant_lev_vol, window_vol)
# df_quantiles = pf.create_quantiles(df, quant_dlev, quant_lev, quant_intan)
pf.fm_ret_lev(df_quantiles)
pf.create_dlev_intan_port(df_quantiles, quant_dlev, quant_intan)
pf.create_dlev_pd_port(df_quantiles, quant_dlev, quant_pd)
pf.create_dlev_lev_port(df_quantiles, quant_dlev, quant_intan, quant_lev, intan_subsample)
# pf.create_dlev_pd_intan_port(df_quantiles, quant_dlev, quant_intan, quant_pd, intan_subsample)
