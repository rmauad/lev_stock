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

# df, port_ff25_pivot = dc.load_fm(redo = False)
df = dc.load_fm(redo = False)
port_ff25_pivot = df

betas_df = pd.read_pickle('../data/pickle/df_betas.pkl')

# df.set_index(['GVKEY', 'year_month'], inplace=True)

# df = df.reset_index()
# df = (df
#       .rename(columns={'GVKEY': 'gvkey'}))
#     #   .drop(columns=['RET_lead1'])
# df.set_index(['gvkey', 'year_month'], inplace=True)

#######################################
quant_dlev = 5
quant_intan = 3
quant_lev = 4
quant_kkr = 5
quant_pd = 5 # probability of default (Merton model)
quant_size = 2 # keep at 2
quant_bm = 3 # keep at 3
intan_subsample = 1 # 1 for low intangibles, quant_intan for high.
pd_subsample = 'lpd' # 'lpd' for low PD, 'hpd' for high PD.
quant_lev_vol = 0
window_vol = 12
subsample = 'lint' # 'all' for all firms, 'hint' for high intangibles, 'lint' for low intangibles.
window_fac_betas = 24 # window for factor betas (24 months)
model_factor = 'ff6' # 'capm', 'ff3', 'ff5', 'ff6'
#######################################

# Print paper tables (summary statistics, Fama-MacBeth regressions, etc.)
# fig = ist.plot_intan_pd(df)
# plt.savefig(figfolder + 'intan_pd.pdf', format='pdf', bbox_inches='tight')
# fig = ist.plot_pd(df_quantiles)
# plt.savefig(figfolder + 'pd.pdf', format='pdf', bbox_inches='tight')


# Motivation (finding firms with high intangible capital, low dlev, and high returns)
# Evoka Pharma GVKEY = 18173
# df_filter = df.reset_index()
# df_filter = df_filter[df_filter['year_month'] >= '2013-01']
# sorted_df = df_filter.sort_values(by=['dlev', 'ret_lead1', 'intan_epk_at'], 
#                            ascending=[True, False, False])
# sorted_df[['year_month', 'GVKEY', 'dlev', 'ret_lead1', 'intan_epk_at']].head(50)

# df_filtered = ist.filter_data(df)
df_quantiles = ist.create_quantiles(df, quant_dlev, quant_intan, quant_lev, quant_kkr, quant_pd, quant_size, quant_bm, quant_lev_vol, window_vol)

# pf.create_sum_stats(df_quantiles)

########################################
# Spanning tests and factor regressions
########################################

df_factor = pf.idl_factor(df_quantiles, quant_intan, quant_dlev, quant_size, quant_bm)
# df_factor_all = pf.ff_factor(df_factor, quant_intan, quant_dlev, quant_size, quant_bm)
# pf.spanning_tests(df_factor_all, filename = f'spanning_test_{model_factor}.tex')
pf.reg_factor(df_factor, quant_intan, quant_dlev, quant_lev, quant_size, quant_bm, window_fac_betas, model_factor, port_ff25_pivot, filename = f'reg_factor_{model_factor}.tex')


betas_df = pf.fama_macbeth_regression(df_quantiles)
# betas_df.to_pickle('../data/pickle/df_betas.pkl')

# results = pf.fama_macbeth_second_stage(df_quantiles, betas_df)
# print(results)

# df_quantiles = pf.create_quantiles(df, quant_dlev, quant_lev, quant_intan)
# pf.create_lev_pd_intan(df_quantiles, quant_lev) #pd and intan by lev quantiles
# # pf.create_intan_lev_pd(df_quantiles, quant_intan) #pd and lev by intan quantiles

# # pf.create_dlev_lev_pd(df_quantiles, quant_dlev, quant_lev)
# # pf.create_dlev_lev_intan(df_quantiles, quant_dlev, quant_lev)

#########################################################################
# Fama MacBeth regressions - main table in the paper
#########################################################################

# filename = f'fm_ret_dlev_{subsample}.tex'
# pf.fm_ret_lev(df_quantiles, quant_intan, quant_lev, subsample, filename = filename)

#########################################################################
# Tables leverage change and intangible/assets by probability of default
#########################################################################

# filename = f'dlev_intan_{pd_subsample}.tex'
# pf.create_dlev_intan_port(df_factor, quant_dlev, quant_intan, quant_pd, pd_subsample, filename = filename)

# pf.create_dlev_pd_port(df_quantiles, quant_dlev, quant_pd)

#########################################################################
# Tables leverage change and intangible/assets by probability of default
#########################################################################

# filename = f'dlev_lev_{subsample}.tex'
# pf.create_dlev_lev_port(df_quantiles, quant_dlev, quant_intan, quant_lev, quant_pd, subsample, filename = filename)

# filename = f'dlev_table_{subsample}.tex'
# pf.create_dlev_table(df_quantiles, quant_dlev, quant_intan, quant_lev, quant_pd, subsample, filename = filename)

# # pf.create_dlev_pd_intan_port(df_quantiles, quant_dlev, quant_intan, quant_pd, intan_subsample)

