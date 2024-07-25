# Defines all functions to create the portfolio strategies.
import pandas as pd
import numpy as np


def create_quantiles(df, quant_dlev, quant_intan, quant_lev, quant_lev_vol, window_vol):
    df_copy = df.copy()
    df_copy['lev_vol'] = df_copy.groupby('gvkey')['debt_at'].transform(lambda x: x.rolling(window_vol).std())

    np.random.seed(42)
    smallest_non_zero = df_copy.loc[df_copy['d_debt_at'] > 0, 'd_debt_at'].min()
    epsilon = np.random.uniform(0, smallest_non_zero * 0.1, size=len(df_copy))

    smallest_non_zero_lev = df_copy.loc[df_copy['debt_at'] > 0, 'debt_at'].min()
    epsilon_lev = np.random.uniform(0, smallest_non_zero_lev * 0.1, size=len(df_copy))

######################################################################
# Most observations are zero because I calculate monthly change
# and the Compustat debt is quarterly. See numbers below and explain
# them in the paper (this is why I create the adjusted variables below)
######################################################################

# Replace zeros with these small random numbers (otherwise, qcut will not work)
    df_copy['d_debt_at_adj'] = df_copy['d_debt_at']
    df_copy['debt_at_adj'] = df_copy['debt_at']
    df_copy['lev_vol_adj'] = df_copy['lev_vol']
    df_copy.loc[df_copy['d_debt_at'] == 0, 'd_debt_at_adj'] = epsilon[df_copy['d_debt_at'] == 0]
    df_copy.loc[df_copy['debt_at'] == 0, 'debt_at_adj'] = epsilon_lev[df_copy['debt_at'] == 0]
    # df_copy.loc[df_copy['lev_vol'] == 0, 'lev_vol_adj'] = epsilon_lev_vol[df_copy['lev_vol'] == 0]

# df.head(50)
    # Define columns and their corresponding quantile counts and names
    quantile_info = {
        'dlev': (quant_dlev, f'd_debt_at_{quant_dlev}'),
        'intan_epk_at': (quant_intan, f'intan_at_{quant_intan}'),
        'lev': (quant_lev, f'debt_at_{quant_lev}'),
        }
    
    quantile_lev_vol_info = {
        'lev_vol': (quant_lev_vol, f'lev_vol_{quant_lev_vol}')
    }

    # Create quantiles based on the provided inputs
    for column, (quant_count, quant_name) in quantile_info.items():
        df_copy[quant_name] = df_copy.groupby('year_month')[column].transform(
            lambda x: pd.qcut(x, quant_count, labels=range(1, quant_count + 1))
        )
    
    if quant_lev_vol > 0:
        # df_copy['lev_vol_1'] = df_copy.groupby('year_month')['lev_vol'].transform(
        #     lambda x: pd.cut(x, bins = [0, 0.00000000001, 0.2, 1], labels = False, include_lowest = True)
        # )
        df_copy = df_copy[df_copy['lev_vol'] > 0]
        for column, (quant_count, quant_name) in quantile_lev_vol_info.items():
            df_copy[quant_name] = df_copy.groupby('year_month')[column].transform(
                lambda x: pd.qcut(x, quant_count, labels=range(1, quant_count + 1))
        )
        return df_copy
    else:
        return df_copy

def create_portfolios(df, quant_dlev, quant_intan, quant_lev, double_strat):
    df_copy = df.copy()

    # Define the names of the quantile columns based on the input quantile values
    quant_dlev_col = f'd_debt_at_{quant_dlev}'
    quant_intan_col = f'intan_at_{quant_intan}'
    quant_lev_col = f'debt_at_{quant_lev}'
    # quant_lev_vol_col = f'lev_vol_{quant_lev_vol}'
      
    df_copy['long'] = np.where((df_copy[quant_dlev_col] == 1), 1, 0)
    df_copy['short'] = np.where((df_copy[quant_dlev_col] == quant_dlev), 1, 0)
    
    df_copy['long_hlev'] = np.where((df_copy[quant_dlev_col] == 1) & (df_copy[quant_lev_col] == quant_lev), 1, 0)
    df_copy['short_hlev'] = np.where((df_copy[quant_dlev_col] == quant_dlev) & (df_copy[quant_lev_col] == quant_lev), 1, 0)
    
    if double_strat == 0:
        df_copy['long_hint'] = np.where((df_copy[quant_dlev_col] == 1) & (df_copy[quant_intan_col] == quant_intan), 1, 0)
        df_copy['short_hint'] = np.where((df_copy[quant_dlev_col] == quant_dlev) & (df_copy[quant_intan_col] == quant_intan), 1, 0)
        df_copy['long_lint'] = np.where((df_copy[quant_dlev_col] == 1) & (df_copy[quant_intan_col] == 1), 1, 0)
        df_copy['short_lint'] = np.where((df_copy[quant_dlev_col] == quant_dlev) & (df_copy[quant_intan_col] == 1), 1, 0)
        
    else:
        df_copy['long_hint'] = np.where((df_copy[quant_dlev_col] == 1) & (df_copy[quant_lev_col] == quant_lev) & (df_copy[quant_intan_col] == quant_intan), 1, 0)
        df_copy['short_hint'] = np.where((df_copy[quant_dlev_col] == quant_dlev) & (df_copy[quant_lev_col] == quant_lev) & (df_copy[quant_intan_col] == quant_intan), 1, 0)
        df_copy['long_lint'] = np.where((df_copy[quant_dlev_col] == 1) & (df_copy[quant_lev_col] == quant_lev) & (df_copy[quant_intan_col] == 1), 1, 0)
        df_copy['short_lint'] = np.where((df_copy[quant_dlev_col] == quant_dlev) & (df_copy[quant_lev_col] == quant_lev) & (df_copy[quant_intan_col] == 1), 1, 0)
        
    return df_copy

# Calculating weighted returns for the portfolios
# def weighted_returns(df, holding_period, all_stocks, value_weight):
#     df_copy = df.copy()
#     # Calculate weights

#     if all_stocks == 1:
#         df_copy['long_weight'] = df_copy['long'] / df_copy.groupby('year_month')['long'].transform('sum')
#         df_copy['short_weight'] = df_copy['short'] / df_copy.groupby('year_month')['short'].transform('sum')
    
#     # Calculate weighted returns
#         for period in range(1, holding_period + 1):
#             ret_col = f'ret_lead{period}'
#             df_copy[ret_col] = df_copy.groupby('gvkey')['ret'].shift(-period)
#             weighted_ret_col = f'weighted_strat_return_lead{period}'
#             df_copy[weighted_ret_col] = df_copy[ret_col] * (df_copy['long_weight'] - df_copy['short_weight'])
    
#     else:
#         if value_weight == 1:
#             df_copy['long_hint_at'] = df_copy['long_hint'] * df_copy['atq']
#             df_copy['short_hint_at'] = df_copy['short_hint'] * df_copy['atq']
#             df_copy['long_lint_at'] = df_copy['long_lint'] * df_copy['atq']
#             df_copy['short_lint_at'] = df_copy['short_lint'] * df_copy['atq']
            
#             df_copy['long_weight_hint'] = df_copy['long_hint_at'] / df_copy.groupby('year_month')['long_hint_at'].transform('sum')
#             df_copy['short_weight_hint'] = df_copy['short_hint_at'] / df_copy.groupby('year_month')['short_hint_at'].transform('sum')
    
#             df_copy['long_weight_lint'] = df_copy['long_lint_at'] / df_copy.groupby('year_month')['long_lint_at'].transform('sum')
#             df_copy['short_weight_lint'] = df_copy['short_lint_at'] / df_copy.groupby('year_month')['short_lint_at'].transform('sum')
    
#             # Calculate weighted returns
#             for period in range(1, holding_period + 1):
#                 ret_col = f'ret_lead{period}'
#                 df_copy[ret_col] = df_copy.groupby('gvkey')['ret'].shift(-period)
#                 weighted_ret_col_hint = f'weighted_strat_return_hint_lead{period}'
#                 weighted_ret_col_lint = f'weighted_strat_return_lint_lead{period}'
#                 df_copy[weighted_ret_col_hint] = df_copy[ret_col] * (df_copy['long_weight_hint'] - df_copy['short_weight_hint'])
#                 df_copy[weighted_ret_col_lint] = df_copy[ret_col] * (df_copy['long_weight_lint'] - df_copy['short_weight_lint'])
                
            
#         else:    
#             df_copy['long_weight_hint'] = df_copy['long_hint'] / df_copy.groupby('year_month')['long_hint'].transform('sum')
#             df_copy['short_weight_hint'] = df_copy['short_hint'] / df_copy.groupby('year_month')['short_hint'].transform('sum')
    
#             df_copy['long_weight_lint'] = df_copy['long_lint'] / df_copy.groupby('year_month')['long_lint'].transform('sum')
#             df_copy['short_weight_lint'] = df_copy['short_lint'] / df_copy.groupby('year_month')['short_lint'].transform('sum')
    
#     # Calculate weighted returns
#             for period in range(1, holding_period + 1):
#                 ret_col = f'ret_lead{period}'
#                 df_copy[ret_col] = df_copy.groupby('gvkey')['ret'].shift(-period)
#                 weighted_ret_col_hint = f'weighted_strat_return_hint_lead{period}'
#                 weighted_ret_col_lint = f'weighted_strat_return_lint_lead{period}'
#                 df_copy[weighted_ret_col_hint] = df_copy[ret_col] * (df_copy['long_weight_hint'] - df_copy['short_weight_hint'])
#                 df_copy[weighted_ret_col_lint] = df_copy[ret_col] * (df_copy['long_weight_lint'] - df_copy['short_weight_lint'])
    
#     return df_copy

def weighted_returns(df, holding_period, value_weight):
    df_copy = df.copy()
    # Calculate weights

    if value_weight == 1:
        df_copy['long_at'] = df_copy['long'] * df_copy['atq']
        df_copy['short_at'] = df_copy['short'] * df_copy['atq']
        df_copy['long_hlev_at'] = df_copy['long_hlev'] * df_copy['atq']
        df_copy['short_hlev_at'] = df_copy['short_hlev'] * df_copy['atq']
        df_copy['long_hint_at'] = df_copy['long_hint'] * df_copy['atq']
        df_copy['short_hint_at'] = df_copy['short_hint'] * df_copy['atq']
        df_copy['long_lint_at'] = df_copy['long_lint'] * df_copy['atq']
        df_copy['short_lint_at'] = df_copy['short_lint'] * df_copy['atq']        
            
        df_copy['long_weight'] = df_copy['long_at'] / df_copy.groupby('year_month')['long_at'].transform('sum')
        df_copy['short_weight'] = df_copy['short_at'] / df_copy.groupby('year_month')['short_at'].transform('sum')
        df_copy['long_weight_hlev'] = df_copy['long_hlev_at'] / df_copy.groupby('year_month')['long_hlev_at'].transform('sum')
        df_copy['short_weight_hlev'] = df_copy['short_hlev_at'] / df_copy.groupby('year_month')['short_hlev_at'].transform('sum')
        df_copy['long_weight_hint'] = df_copy['long_hint_at'] / df_copy.groupby('year_month')['long_hint_at'].transform('sum')
        df_copy['short_weight_hint'] = df_copy['short_hint_at'] / df_copy.groupby('year_month')['short_hint_at'].transform('sum')
        df_copy['long_weight_lint'] = df_copy['long_lint_at'] / df_copy.groupby('year_month')['long_lint_at'].transform('sum')
        df_copy['short_weight_lint'] = df_copy['short_lint_at'] / df_copy.groupby('year_month')['short_lint_at'].transform('sum')

    else:
        df_copy['long_weight'] = df_copy['long'] / df_copy.groupby('year_month')['long'].transform('sum')
        df_copy['short_weight'] = df_copy['short'] / df_copy.groupby('year_month')['short'].transform('sum')
        df_copy['long_weight_hlev'] = df_copy['long_hlev'] / df_copy.groupby('year_month')['long_hlev'].transform('sum')
        df_copy['short_weight_hlev'] = df_copy['short_hlev'] / df_copy.groupby('year_month')['short_hlev'].transform('sum')
        df_copy['long_weight_hint'] = df_copy['long_hint'] / df_copy.groupby('year_month')['long_hint'].transform('sum')
        df_copy['short_weight_hint'] = df_copy['short_hint'] / df_copy.groupby('year_month')['short_hint'].transform('sum')
        df_copy['long_weight_lint'] = df_copy['long_lint'] / df_copy.groupby('year_month')['long_lint'].transform('sum')
        df_copy['short_weight_lint'] = df_copy['short_lint'] / df_copy.groupby('year_month')['short_lint'].transform('sum')

    # Calculate weighted returns
    for period in range(1, holding_period + 1):
        ret_col = f'ret_lead{period}'
        df_copy[ret_col] = df_copy.groupby('gvkey')['ret'].shift(-period)
        weighted_ret_col = f'weighted_strat_return_lead{period}'
        weighted_ret_col_hlev = f'weighted_strat_return_hlev_lead{period}'
        weighted_ret_col_hint = f'weighted_strat_return_hint_lead{period}'
        weighted_ret_col_lint = f'weighted_strat_return_lint_lead{period}'
        df_copy[weighted_ret_col] = df_copy[ret_col] * (df_copy['long_weight'] - df_copy['short_weight'])
        df_copy[weighted_ret_col_hlev] = df_copy[ret_col] * (df_copy['long_weight_hlev'] - df_copy['short_weight_hlev'])        
        df_copy[weighted_ret_col_hint] = df_copy[ret_col] * (df_copy['long_weight_hint'] - df_copy['short_weight_hint'])
        df_copy[weighted_ret_col_lint] = df_copy[ret_col] * (df_copy['long_weight_lint'] - df_copy['short_weight_lint'])
        
    return df_copy


# Aggregating weighted returns
# def agg_weighted_returns(df, holding_period, all_stocks):
#     aggregation_dict = {}

#     if all_stocks == 1:
#         for period in range(1, holding_period + 1):
#             weighted_ret_col = f'weighted_strat_return_lead{period}'
#             aggregation_dict[weighted_ret_col] = 'sum'
            
#         df_port_ret_agg = df.groupby('year_month').agg(aggregation_dict).reset_index()
#     else:
#         for period in range(1, holding_period + 1):
#             weighted_ret_col_hint = f'weighted_strat_return_hint_lead{period}'
#             weighted_ret_col_lint = f'weighted_strat_return_lint_lead{period}'
#             aggregation_dict[weighted_ret_col_hint] = 'sum'
#             aggregation_dict[weighted_ret_col_lint] = 'sum'
            
#         df_port_ret_agg = df.groupby('year_month').agg(aggregation_dict).reset_index()
    
#     return df_port_ret_agg

def agg_weighted_returns(df, holding_period):
    aggregation_dict = {}
    
    for period in range(1, holding_period + 1):
        weighted_ret_col = f'weighted_strat_return_lead{period}'
        weighted_ret_col_hlev = f'weighted_strat_return_hlev_lead{period}'
        weighted_ret_col_hint = f'weighted_strat_return_hint_lead{period}'
        weighted_ret_col_lint = f'weighted_strat_return_lint_lead{period}'
        aggregation_dict[weighted_ret_col] = 'sum'
        aggregation_dict[weighted_ret_col_hlev] = 'sum'        
        aggregation_dict[weighted_ret_col_hint] = 'sum'
        aggregation_dict[weighted_ret_col_lint] = 'sum'
        
    df_port_ret_agg = df.groupby('year_month').agg(aggregation_dict).reset_index()
        
    return df_port_ret_agg


# def avr_port_holding_period(df, holding_period, all_stocks):
#     df_port_ret_agg = df.copy()
    
#     if all_stocks == 1:
#         for i in range(1, df_port_ret_agg.shape[0] - holding_period + 2):
#             port_col = f'port_hint{i}'
#             df_port_ret_agg[port_col] = np.nan
#             for period in range(holding_period):
#                 lead_col = f'weighted_strat_return_lead{period + 1}'
#                 if i + period < df_port_ret_agg.shape[0]:
#                     df_port_ret_agg.loc[i + period, port_col] = df_port_ret_agg.loc[i + period, lead_col]  
                
#     else:      
#         for i in range(1, df_port_ret_agg.shape[0] - holding_period + 2):
#             port_col = f'port_hint{i}'
#             df_port_ret_agg[port_col] = np.nan
#             for period in range(holding_period):
#                 lead_col = f'weighted_strat_return_hint_lead{period + 1}'
#                 if i + period < df_port_ret_agg.shape[0]:
#                     df_port_ret_agg.loc[i + period, port_col] = df_port_ret_agg.loc[i + period, lead_col]
                
#         for i in range(1, df_port_ret_agg.shape[0] - holding_period + 2):
#             port_col = f'port_lint{i}'
#             df_port_ret_agg[port_col] = np.nan
#             for period in range(holding_period):
#                 lead_col = f'weighted_strat_return_lint_lead{period + 1}'
#                 if i + period < df_port_ret_agg.shape[0]:
#                     df_port_ret_agg.loc[i + period, port_col] = df_port_ret_agg.loc[i + period, lead_col]

#     return df_port_ret_agg

def avr_port_holding_period(df, holding_period):
    df_port_ret_agg = df.copy()
    
    for i in range(1, df_port_ret_agg.shape[0] - holding_period + 2):
        port_col = f'port{i}'
        port_hlev_col = f'port_hlev{i}'
        port_hint_col = f'port_hint{i}'
        port_lint_col = f'port_lint{i}'
        df_port_ret_agg[port_col] = np.nan
        df_port_ret_agg[port_hlev_col] = np.nan
        df_port_ret_agg[port_hint_col] = np.nan
        df_port_ret_agg[port_lint_col] = np.nan
        for period in range(holding_period):
            lead_col = f'weighted_strat_return_lead{period + 1}'
            lead_hlev_col = f'weighted_strat_return_hlev_lead{period + 1}'
            lead_hint_col = f'weighted_strat_return_hint_lead{period + 1}'
            lead_lint_col = f'weighted_strat_return_lint_lead{period + 1}'
            if i + period < df_port_ret_agg.shape[0]:
                df_port_ret_agg.loc[i + period, port_col] = df_port_ret_agg.loc[i + period, lead_col]  
                df_port_ret_agg.loc[i + period, port_hlev_col] = df_port_ret_agg.loc[i + period, lead_hlev_col]  
                df_port_ret_agg.loc[i + period, port_hint_col] = df_port_ret_agg.loc[i + period, lead_hint_col]  
                df_port_ret_agg.loc[i + period, port_lint_col] = df_port_ret_agg.loc[i + period, lead_lint_col]  

    return df_port_ret_agg


# def calculate_annualized_sharpe_ratio(ret_rf):
#     ret_rf_copy = ret_rf.copy()
#     """Calculate the annualized Sharpe ratio for a series of monthly returns."""
#     # Convert monthly risk-free rate to the same frequency as returns
#     rf_monthly = (1 + ret_rf_copy['rf_annual']/100) ** (1/12) - 1
#     ret_rf_copy = ret_rf_copy.assign(rf_monthly=rf_monthly)
#     # excess_returns = ret_rf[['strat_hint_ret', 'strat_lint_ret', 'sp500_ret']] - rf_monthly
#     excess_returns = ret_rf_copy[['strat_hint_ret', 'strat_lint_ret', 'sp500_ret']].subtract(ret_rf_copy['rf_monthly'], axis=0)
#     # excess_returns_dates = pd.concat([ret_rf_copy['year_month'], excess_returns], axis=1)
#     dates = ret_rf_copy['year_month']

#     # Annualize the mean and standard deviation of excess returns
#     mean_excess_return = np.mean(excess_returns)
#     std_excess_return = np.std(excess_returns)
    
#     ret_rf_copy = (mean_excess_return * 12) / (std_excess_return * np.sqrt(12))
#     return ret_rf_copy, excess_returns, dates

def calculate_annualized_sharpe_ratio(ret_rf):
    ret_rf_copy = ret_rf.copy()
    """Calculate the annualized Sharpe ratio for a series of monthly returns."""
    # Convert monthly risk-free rate to the same frequency as returns

    excess_returns = ret_rf_copy[['strat_ret', 'strat_hlev_ret', 'strat_hint_ret', 'strat_lint_ret']].subtract(ret_rf_copy['rf']/100, axis=0)
    # excess_returns_dates = pd.concat([ret_rf_copy['year_month'], excess_returns], axis=1)
    dates = ret_rf_copy['year_month']

    # Annualize the mean and standard deviation of excess returns
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns)
    
    ret_rf_copy = (mean_excess_return * 12) / (std_excess_return * np.sqrt(12))
    return ret_rf_copy, excess_returns, dates


def rolling_sharpe_ratio(excess_returns, window, dates):
    """Calculate the rolling Sharpe ratio for a series of excess returns using a rolling window."""
    # rolling_sharpe = excess_returns.rolling(window=window).apply(
    #     lambda x: (np.mean(x) * 12) / (np.std(x) * np.sqrt(12)), raw=False
    # )
    
    # Calculate rolling mean and std
    rolling_mean = excess_returns.rolling(window=window).mean()
    rolling_std = excess_returns.rolling(window=window).std()

    # Annualize the mean and standard deviation
    annualized_mean = rolling_mean * 12
    annualized_std = rolling_std * np.sqrt(12)

    # Calculate the rolling Sharpe ratio
    rolling_sharpe = annualized_mean / annualized_std

    # Adjust dates to exclude the initial window size
    adjusted_dates = dates[window - 1:]
    adjusted_sharpe = rolling_sharpe[window - 1:]

    result_df = pd.DataFrame({'year_month': adjusted_dates.values})
    
    # Concatenate the adjusted_sharpe to the result DataFrame
    for col in adjusted_sharpe.columns:
        result_df[col] = adjusted_sharpe[col].values
    
    return result_df


    # rolling_sharpe_df = pd.DataFrame({'Date': adjusted_dates, 'Rolling_Sharpe_Ratio': adjusted_sharpe.values})
    
    # return rolling_sharpe_df

def create_factors(df):
    df_copy = df.copy()
    df_copy['dlev'] = df_copy.groupby('gvkey')['ret'].transform(lambda x: x.rolling(12).sum())