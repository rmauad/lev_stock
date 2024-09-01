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

df = dc.load_fm(redo = False)

df.set_index(['GVKEY', 'year_month'], inplace=True)

#######################################
quant_dlev = 5
quant_intan = 5
quant_lev = 4
quant_kkr = 5
quant_pd = 5 # probability of default (Merton model)
intan_subsample = 1 # 1 for low intangibles, quant_intan for high.
quant_lev_vol = 0
window_vol = 12
subsample = 'lint' # 'all' for all firms, 'hint' for high intangibles, 'lint' for low intangibles.
#######################################

df_quantiles = ist.create_quantiles(df, quant_dlev, quant_intan, quant_lev, quant_kkr, quant_pd, quant_lev_vol, window_vol)
filename = f'ts_dlev_{subsample}.tex'
pf.table_ff_dlev(df_quantiles, quant_dlev, quant_intan, subsample, filename = filename)




