import pandas as pd
# from tabulate import tabulate
import sys
sys.path.append('code/lev_stock/portfolio/')
import portfolio_functions as pf
import data_treat_port_functions as dtpf

df = pd.read_feather('data/feather/df_fm.feather') # from prep_fm.py

#######################################
quant_intan = 3 # 1 for low, 3 for high

#######################################

# Print paper tables (summary statistics, Fama-MacBeth regressions, etc.)
pf.create_sum_stats(df)
df_intan_dummy = dtpf.create_intan_dummy(df)
pf.fm_ret_lev(df_intan_dummy)
df_quantiles = dtpf.create_quantiles(df_intan_dummy)
pf.create_dlev_intan_port(df_quantiles)
pf.create_dlev_lev_port(df_quantiles, quant_intan)
