# Defines all functions to create the portfolio strategies.
import pandas as pd
import numpy as np
from .announce_execution import announce_execution

@announce_execution
def calc_avr_portfolio(df):

    df_copy = df.copy()
    # Create a new column 'average_port' by taking the mean across all 'port' columns
    port_columns = [col for col in df_copy.columns if col.startswith('port')]
    df_copy['strat_ret'] = df_copy[port_columns].mean(axis=1, skipna=True)
    port_hlev_columns = [col for col in df_copy.columns if col.startswith('port_hlev')]
    df_copy['strat_hlev_ret'] = df_copy[port_hlev_columns].mean(axis=1, skipna=True)
    port_hint_columns = [col for col in df_copy.columns if col.startswith('port_hint')]
    df_copy['strat_hint_ret'] = df_copy[port_hint_columns].mean(axis=1, skipna=True)
    port_lint_columns = [col for col in df_copy.columns if col.startswith('port_lint')]
    df_copy['strat_lint_ret'] = df_copy[port_lint_columns].mean(axis=1, skipna=True)
    df_strat_ret = df_copy[['year_month', 'strat_ret', 'strat_hlev_ret', 'strat_hint_ret', 'strat_lint_ret']]

    return df_strat_ret


# def merge_df_rf(df, rf):
#     df_copy = df.copy()
#     df_copy = df_copy.assign(year_month = lambda x: x['year_month'].dt.to_period('M'))
    
#     rf_rate = (rf
#       .assign(date_ret = pd.to_datetime(rf['DATE'], format='%Y-%m-%d', errors = "coerce"))
#       .assign(year_month = lambda x: x['date_ret'].dt.to_period('M'))
#       .assign(rf_annual = lambda x: pd.to_numeric(x['TB3MS'], errors = "coerce"))
#       .drop(columns= ['DATE', 'date_ret', 'TB3MS'])
#       .query('year_month >= "1980-01" and year_month <= "2020-12"')
#       .reset_index(drop = True)
#       )

#     returns = df_copy[['year_month', 'strat_hint_ret', 'strat_lint_ret', 'sp500_ret']]
#     # window = 60
#     rf_annual = rf_rate[['year_month', 'rf_annual']]
#     # ret_rf.head(50)
#     ret_rf = pd.merge(returns, rf_annual, on = 'year_month', how = 'left')

#     return ret_rf
@announce_execution
def merge_df_rf(df, rf):
    df_copy = df.copy()
    df_copy = df_copy.reset_index()
    rf_agg = rf.groupby('year_month').first().reset_index()

    returns = df_copy[['year_month', 'strat_ret', 'strat_hlev_ret', 'strat_hint_ret', 'strat_lint_ret']]

    ret_rf = pd.merge(returns, rf_agg, on = 'year_month', how = 'left')

    return ret_rf
  
def merge_ff_factors(df, ff):
    df_copy = df.copy()
    # Merge the Fama-French factors
    # df_copy['year_month'] = df_copy['date_ret'].dt.to_period('M')
    df_copy = df_copy.reset_index()
    df_copy['year_month'] = df_copy['year_month'].dt.to_period('M')

    ff = (ff
          .assign(date = pd.to_datetime(ff['date'], format='%Y%m'))
          # change frequency to monthly
          .assign(year_month = lambda x: x['date'].dt.to_period('M'))
    )
    df_copy = pd.merge(df_copy, ff, on = 'year_month', how = 'left')
    
    return df_copy