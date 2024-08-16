# Defines all functions to create the portfolio strategies.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .announce_execution import announce_execution


def plot_intan_pd(df):
    plt.rcParams.update({
    'font.size': 22,           # General font size
    'axes.titlesize': 24,      # Title font size
    'axes.labelsize': 18,      # Axis label font size
    'xtick.labelsize': 18,     # X-tick label font size
    'ytick.labelsize': 18,     # Y-tick label font size
    'legend.fontsize': 18,     # Legend font size
    'figure.titlesize': 24     # Figure title font size
    })


    df_copy = df.copy()
    # df_copy = (df_copy
    #             .assign(intan_at_avg = df_copy.groupby('year_month')['intan_epk_at'].transform('mean'))
    #             .assign(pd_at_avg = df_copy.groupby('year_month')['default_probability'].transform('mean'))
    #             )
    
    df_copy = (df_copy
               .assign(weighted_intan = df_copy['intan_epk_at'] * df_copy['atq'])
               .assign(weighted_pd = df_copy['default_probability'] * df_copy['atq'])
               .assign(sum_weights = df_copy.groupby('year_month')['atq'].transform('sum'))
               .assign(intan_at_avg = lambda x: x.groupby('year_month')['weighted_intan'].transform('sum') / x['sum_weights'])
               .assign(pd_at_avg = lambda x: x.groupby('year_month')['weighted_pd'].transform('sum') / x['sum_weights'])
               )
    
    df_copy = df_copy.reset_index()
    df_avg = df_copy.drop_duplicates(subset=['year_month'])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_avg['intan_at_avg'], df_avg['pd_at_avg'])
    ax.set_title('Intangible/Assets and Probability of Default')
    ax.set_xlabel('Intangible/Assets')
    ax.set_ylabel('Average probability of default')
    ax.grid(True)
    
    fig.tight_layout()  # Automatically adjust layout to fit within figure area

    return fig

def plot_pd(df):
    plt.rcParams.update({
    'font.size': 22,           # General font size
    'axes.titlesize': 24,      # Title font size
    'axes.labelsize': 18,      # Axis label font size
    'xtick.labelsize': 18,     # X-tick label font size
    'ytick.labelsize': 18,     # Y-tick label font size
    'legend.fontsize': 18,     # Legend font size
    'figure.titlesize': 24     # Figure title font size
    })


    df_copy = df.copy()
    # df_copy = (df_copy
    #             .assign(intan_at_avg = df_copy.groupby('year_month')['intan_epk_at'].transform('mean'))
    #             .assign(pd_at_avg = df_copy.groupby('year_month')['default_probability'].transform('mean'))
    #             )
    
    df_copy = (df_copy
               .assign(weighted_pd = df_copy['default_probability'] * df_copy['atq'])
               .assign(sum_weights = df_copy.groupby('year_month')['atq'].transform('sum'))
               .assign(pd_at_avg = lambda x: x.groupby('year_month')['weighted_pd'].transform('sum') / x['sum_weights'])
               )
    
    df_copy = df_copy.reset_index()
    df_avg = df_copy.drop_duplicates(subset=['year_month'])
    df_avg['year_month_dt'] = df_avg['year_month'].dt.to_timestamp() 
    df_avg.sort_values('year_month_dt', inplace=True)
    # df_avg = df_avg[df_avg['year_month_dt'] <= '1999-01-01']

    # Example recession dates
    recession_dates = [
        ('1980-01-01', '1980-07-01'),
        ('1981-07-01', '1982-11-01'),
        ('1990-07-01', '1990-10-01'),
        ('2001-04-01', '2001-11-01'),
        ('2007-12-01', '2009-06-01'),
        ('2007-12-01', '2009-06-01'),
        ('2020-02-01', '2020-04-01'),
        # Add more recessions as needed
        ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_avg['year_month_dt'], df_avg['pd_at_avg'])  # 'marker' argument is optional, adds points on the line
    ax.set_title('Probability of Default (Merton Model)')
    ax.set_xlabel('Period')
    ax.set_ylabel('Average probability of default')
    ax.grid(True)
    
    # Adding the shaded areas for recessions
    for start_date, end_date in recession_dates:
        ax.axvspan(pd.to_datetime(start_date), pd.to_datetime(end_date), color='gray', alpha=0.5)


    fig.tight_layout()  # Automatically adjust layout to fit within figure area

    return fig

def plot_returns_mkt(df, sp500, all_stocks):
    plt.rcParams.update({
        'font.size': 22,           # General font size
        'axes.titlesize': 24,      # Title font size
        'axes.labelsize': 18,      # Axis label font size
        'xtick.labelsize': 18,     # X-tick label font size
        'ytick.labelsize': 18,     # Y-tick label font size
        'legend.fontsize': 18,     # Legend font size
        'figure.titlesize': 24     # Figure title font size
        })

    if all_stocks == 1:
        df_copy = df.copy()
        df_copy = (df_copy
                   .assign(strat_ret_hint_cum = (1 + df_copy['strat_ret']).cumprod())
                   )
    
    # Normalize cumulative returns such that they start at 1
        df_copy['strat_ret_cum_norm'] = df_copy['strat_ret_cum'] / df_copy['strat_ret_cum'].iloc[1]
    
    # Including the S&P 500 returns
        sp500_long = sp500.melt(id_vars=['Year'], var_name='Month', value_name='Return')
        sp500_long['year_month'] = pd.to_datetime(sp500_long['Year'].astype(str) + '-' + sp500_long['Month'], format='%Y-%b')
        sp500_long = sp500_long.drop(['Year', 'Month'], axis=1)
        sp500_long = sp500_long.sort_values('year_month')
        sp500_long.head(50)
        
        sp500_long.to_excel('sp500_long.xlsx', index = False)
    
        df_sp500 = df_copy.merge(sp500_long, on='year_month', how='inner')
        df_sp500 = (df_sp500
                    .assign(sp500_cum = (1 + df_sp500['Return']).cumprod())
                    .assign(sp500_cum_norm = lambda x: x['sp500_cum'] / x['sp500_cum'].iloc[1])
                    .rename(columns={'Return': 'sp500_ret'})
                    )

# save = df_sp500.to_feather('data/feather/df_strat_ret_sp500.feather')

# Plotting the cumulative returns
        fig, ax = plt.subplots()
        ax = df_sp500.plot(x='year_month', y=['strat_ret_cum_norm', 'sp500_cum_norm'], figsize=(10, 6))
        ax.set_title('Cumulative returns of investment strategies')
        ax.set_ylabel('Cumulative return')
        ax.set_xlabel('Period')
        ax.grid()
        ax.legend(['HML leverage change', 'S&P 500'])
        plt.show()
        
    else:
        df_copy = df.copy()
        df_copy = (df_copy
                   .assign(strat_ret_hint_cum = (1 + df_copy['strat_hint_ret']).cumprod())
                   .assign(strat_ret_lint_cum = (1 + df_copy['strat_lint_ret']).cumprod())
                   )
    
    # Normalize cumulative returns such that they start at 1
        df_copy['strat_ret_hint_cum_norm'] = df_copy['strat_ret_hint_cum'] / df_copy['strat_ret_hint_cum'].iloc[1]
        df_copy['strat_ret_lint_cum_norm'] = df_copy['strat_ret_lint_cum'] / df_copy['strat_ret_lint_cum'].iloc[1]
    
    # Including the S&P 500 returns
        sp500_long = sp500.melt(id_vars=['Year'], var_name='Month', value_name='Return')
        sp500_long['year_month'] = pd.to_datetime(sp500_long['Year'].astype(str) + '-' + sp500_long['Month'], format='%Y-%b')
        sp500_long = sp500_long.drop(['Year', 'Month'], axis=1)
        sp500_long = sp500_long.sort_values('year_month')
        # sp500_long.head(50)
    
        df_sp500 = df_copy.merge(sp500_long, on='year_month', how='inner')
        df_sp500 = (df_sp500
                    .assign(sp500_cum = (1 + df_sp500['Return']).cumprod())
                    .assign(sp500_cum_norm = lambda x: x['sp500_cum'] / x['sp500_cum'].iloc[1])
                    .rename(columns={'Return': 'sp500_ret'})
                    )

# save = df_sp500.to_feather('data/feather/df_strat_ret_sp500.feather')

# Plotting the cumulative returns
        fig, ax = plt.subplots()
        ax = df_sp500.plot(x='year_month', y=['strat_ret_hint_cum_norm', 'strat_ret_lint_cum_norm', 'sp500_cum_norm'], figsize=(10, 6))
        ax.set_title('Leverage change investment strategies')
        ax.set_ylabel('Cumulative return')
        ax.set_xlabel('Period')
        ax.grid()
        ax.legend(['High intangible/assets', 'Low intangible/assets', 'S&P 500'],
                  loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
        fig.tight_layout()  # Automatically adjust layout to fit within figure area

        plt.show()
    
        return df_sp500, fig

# def plot_returns(df, all_stocks):
#     plt.rcParams.update({
#         'font.size': 22,           # General font size
#         'axes.titlesize': 24,      # Title font size
#         'axes.labelsize': 18,      # Axis label font size
#         'xtick.labelsize': 18,     # X-tick label font size
#         'ytick.labelsize': 18,     # Y-tick label font size
#         'legend.fontsize': 18,     # Legend font size
#         'figure.titlesize': 24     # Figure title font size
#         })

#     if all_stocks == 1:
#         df_copy = df.copy()
#         df_copy = (df_copy
#                    .assign(strat_ret_hint_cum = (1 + df_copy['strat_ret']).cumprod())
#                    )
    
#     # Normalize cumulative returns such that they start at 1
#         df_copy['strat_ret_cum_norm'] = df_copy['strat_ret_cum'] / df_copy['strat_ret_cum'].iloc[1]
    
#     # Including the S&P 500 returns
#         sp500_long = sp500.melt(id_vars=['Year'], var_name='Month', value_name='Return')
#         sp500_long['year_month'] = pd.to_datetime(sp500_long['Year'].astype(str) + '-' + sp500_long['Month'], format='%Y-%b')
#         sp500_long = sp500_long.drop(['Year', 'Month'], axis=1)
#         sp500_long = sp500_long.sort_values('year_month')
#         sp500_long.head(50)
    
#         df_sp500 = df_copy.merge(sp500_long, on='year_month', how='inner')
#         df_sp500 = (df_sp500
#                     .assign(sp500_cum = (1 + df_sp500['Return']).cumprod())
#                     .assign(sp500_cum_norm = lambda x: x['sp500_cum'] / x['sp500_cum'].iloc[1])
#                     .rename(columns={'Return': 'sp500_ret'})
#                     )

# # save = df_sp500.to_feather('data/feather/df_strat_ret_sp500.feather')

# # Plotting the cumulative returns
#         fig, ax = plt.subplots()
#         ax = df_sp500.plot(x='year_month', y=['strat_ret_cum_norm', 'sp500_cum_norm'], figsize=(10, 6))
#         ax.set_title('Cumulative returns of investment strategies')
#         ax.set_ylabel('Cumulative return')
#         ax.set_xlabel('Period')
#         ax.grid()
#         ax.legend(['HML leverage change', 'S&P 500'])
#         plt.show()
        
#     else:
#         df_copy = df.copy()
#         df_copy = (df_copy
#                    .assign(strat_ret_hint_cum = (1 + df_copy['strat_hint_ret']).cumprod())
#                    .assign(strat_ret_lint_cum = (1 + df_copy['strat_lint_ret']).cumprod())
#                    )
    
#     # Normalize cumulative returns such that they start at 1
#         df_copy['strat_ret_hint_cum_norm'] = df_copy['strat_ret_hint_cum'] / df_copy['strat_ret_hint_cum'].iloc[1]
#         df_copy['strat_ret_lint_cum_norm'] = df_copy['strat_ret_lint_cum'] / df_copy['strat_ret_lint_cum'].iloc[1]
    

# # save = df_sp500.to_feather('data/feather/df_strat_ret_sp500.feather')

# # Plotting the cumulative returns
#         fig, ax = plt.subplots()
#         ax = df_copy.plot(x='year_month', y=['strat_ret_hint_cum_norm', 'strat_ret_lint_cum_norm'], figsize=(10, 6))
#         ax.set_title('Leverage change investment strategies')
#         ax.set_ylabel('Cumulative return')
#         ax.set_xlabel('Period')
#         ax.grid()
#         ax.legend(['High intangible/assets', 'Low intangible/assets'],
#                   loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
#         fig.tight_layout()  # Automatically adjust layout to fit within figure area

#         plt.show()
    
#         return df_copy, fig

@announce_execution
def plot_returns(df, double_strat):
    plt.rcParams.update({
        'font.size': 22,           # General font size
        'axes.titlesize': 24,      # Title font size
        'axes.labelsize': 18,      # Axis label font size
        'xtick.labelsize': 18,     # X-tick label font size
        'ytick.labelsize': 18,     # Y-tick label font size
        'legend.fontsize': 18,     # Legend font size
        'figure.titlesize': 24     # Figure title font size
        })


    df_copy = df.copy()
    df_copy = (df_copy
                .assign(strat_ret_cum = (1 + df_copy['strat_ret']).cumprod())
                .assign(strat_ret_hlev_cum = (1 + df_copy['strat_hlev_ret']).cumprod())
                # .assign(strat_ret_hpd_cum = (1 + df_copy['strat_hpd_ret']).cumprod())               
                .assign(strat_ret_hint_cum = (1 + df_copy['strat_hint_ret']).cumprod())
                .assign(strat_ret_lint_cum = (1 + df_copy['strat_lint_ret']).cumprod())
                )

# Normalize cumulative returns such that they start at 1
    df_copy['strat_ret_cum_norm'] = df_copy['strat_ret_cum'] / df_copy['strat_ret_cum'].iloc[1]
    df_copy['strat_ret_hlev_cum_norm'] = df_copy['strat_ret_hlev_cum'] / df_copy['strat_ret_hlev_cum'].iloc[1]
    # df_copy['strat_ret_hpd_cum_norm'] = df_copy['strat_ret_hpd_cum'] / df_copy['strat_ret_hpd_cum'].iloc[1]
    df_copy['strat_ret_hint_cum_norm'] = df_copy['strat_ret_hint_cum'] / df_copy['strat_ret_hint_cum'].iloc[1]
    df_copy['strat_ret_lint_cum_norm'] = df_copy['strat_ret_lint_cum'] / df_copy['strat_ret_lint_cum'].iloc[1]


# save = df_sp500.to_feather('data/feather/df_strat_ret_sp500.feather')

# Plotting the cumulative returns

    fig, ax = plt.subplots()
    ax = df_copy.plot(x='year_month', y=['strat_ret_cum_norm', 'strat_ret_hlev_cum_norm', 'strat_ret_hint_cum_norm', 'strat_ret_lint_cum_norm'], figsize=(10, 6))
    ax.set_title('Leverage change investment strategies')
    ax.set_ylabel('Cumulative return')
    ax.set_xlabel('Period')
    ax.grid()
    
    if double_strat == 0:
        ax.legend(['All stocks', 'High leverage level', 'High intangible/assets', 'Low intangible/assets'],
                    loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    else:
        ax.legend(['All stocks', 'High leverage level', 'High leverage level and high intangible/assets', 'High leverage level and low intangible/assets'],
                    loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1)
    
    fig.tight_layout()  # Automatically adjust layout to fit within figure area

    # plt.show()

    return df_copy, fig
    
# def plot_sharpe(rolling_sharpe_ratio):
#     plot_sharpe, ax = plt.subplots()
#     ax = rolling_sharpe_ratio.plot(x='year_month', y=['strat_hint_ret', 'strat_lint_ret', 'sp500_ret'], figsize=(10, 6))
#     ax.set_title('Sharpe ratio leverage change investment strategies')
#     ax.set_ylabel('5-year rolling window Sharpe ratio')
#     ax.set_xlabel('Period')
#     ax.grid()
#     ax.legend(['SR High intangible/assets', 'SR Low intangible/assets', 'SR S&P 500'],
#               loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
#     plot_sharpe.tight_layout()  # Automatically adjust layout to fit within figure area

#     plt.show()

#     return plot_sharpe

@announce_execution
def plot_sharpe(rolling_sharpe_ratio, double_strat):
    plot_sharpe, ax = plt.subplots()
    ax = rolling_sharpe_ratio.plot(x='year_month', y=['strat_ret', 'strat_hlev_ret', 'strat_hint_ret', 'strat_lint_ret'], figsize=(10, 6))
    ax.set_title('Sharpe ratio leverage change investment strategies')
    ax.set_ylabel('5-year rolling window Sharpe ratio')
    ax.set_xlabel('Period')
    ax.grid()
    
    if double_strat == 0:
        ax.legend(['SR all stocks', 'SR high leverage level', 'SR high intangible/assets', 'SR low intangible/assets'],
                loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    else:
        ax.legend(['SR all stocks', 'SR high leverage level', 'SR high leverage level and high intangible/assets', 'SR high leverage level and low intangible/assets'],
                loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1)        
    
    plot_sharpe.tight_layout()  # Automatically adjust layout to fit within figure area

    #plt.show()
    
    return plot_sharpe
