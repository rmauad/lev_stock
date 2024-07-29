import pandas as pd
# from tabulate import tabulate
import sys
sys.path.append('code/lev_stock/portfolio/')
sys.path.append('code/lev_stock/invest_strat/')
import portfolio as pf
import datacleaning as dc
import invest_strat as ist
# import portfolio_functions as pf
# import data_treat_port_functions as dtpf

# df = pd.read_feather('data/feather/df_fm.feather') # from prep_fm.py
df = dc.load_fm(redo = False)

#######################################
quant_dlev = 5
quant_intan = 5
quant_lev = 5
intan_subsample = 1 # 1 for low intangibles, quant_intan for high.
quant_lev_vol = 0
window_vol = 12
#######################################

# Print paper tables (summary statistics, Fama-MacBeth regressions, etc.)
pf.create_sum_stats(df)
df_quantiles = ist.create_quantiles(df, quant_dlev, quant_intan, quant_lev, quant_lev_vol, window_vol)
pf.fm_ret_lev(df_quantiles)
pf.create_dlev_intan_port(df_quantiles, quant_dlev, quant_intan)
pf.create_dlev_lev_port(df_quantiles, quant_dlev, quant_intan, quant_lev, intan_subsample)
