import pandas as pd
import numpy as np
from tabulate import tabulate
from linearmodels.panel.model import FamaMacBeth
import statsmodels.api as sm
import sys
sys.path.append('code/lev_stock/portfolio/')
sys.path.append('code/lev_stock/invest_strat/')
import utils as ut


def table_ff_dlev(df, quant_dlev, quant_intan, subsample, filename):
    df_copy = df.copy()
    if subsample == 'hint':
        df_subsample = df_copy[df_copy['quant_intan'] == quant_intan]
    if subsample == 'lint':
            df_subsample = df_copy[df_copy['quant_intan'] == 1]
    else:
        df_subsample = df_copy

    