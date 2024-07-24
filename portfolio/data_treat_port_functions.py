import pandas as pd
import numpy as np


def create_intan_dummy(df):
    df_copy = df.copy()
    df_copy['ter_intan_at'] = df_copy.groupby('year_month')['intan_epk_at'].transform(
        lambda x: pd.qcut(x, 3, labels=[1, 2, 3])
    )

    df_copy = df_copy.assign(hint = df_copy['ter_intan_at'] == 3)
    df_copy = df_copy.assign(lint = df_copy['ter_intan_at'] == 1)
    df_copy['dummyXd_debt_at'] = df_copy['d_debt_at'] * df_copy['hint']

    return df_copy

def create_quantiles(df):
    df_copy = df.copy()
    #################################################################################
    # To generate quintiles of leverage change by month, 
    # I need to deal with the zeros (otherwise, I get "Bin edges must be unique" error)
    # Solution: replace zeros with the smallest non-zero value - espilon
    #################################################################################
    np.random.seed(42)
    smallest_non_zero = df_copy.loc[df_copy['d_debt_at'] > 0, 'd_debt_at'].min()
    print(smallest_non_zero)
    epsilon = np.random.uniform(0, smallest_non_zero * 0.1, size=len(df_copy))

    smallest_non_zero_lev = df_copy.loc[df_copy['debt_at'] > 0, 'debt_at'].min()
    print(smallest_non_zero_lev)
    epsilon_lev = np.random.uniform(0, smallest_non_zero_lev * 0.1, size=len(df_copy))

    # Replace zeros with these small random numbers (otherwise, qcut will not work)
    df_copy['d_debt_at_adj'] = df_copy['d_debt_at']
    df_copy['debt_at_adj'] = df_copy['debt_at']
    df_copy.loc[df_copy['d_debt_at'] == 0, 'd_debt_at_adj'] = epsilon[df_copy['d_debt_at'] == 0]
    df_copy.loc[df_copy['debt_at'] == 0, 'debt_at_adj'] = epsilon_lev[df_copy['debt_at'] == 0]

    # df.head(50)

    df_copy['qui_d_debt_at'] = df_copy.groupby('year_month')['d_debt_at_adj'].transform(
        lambda x: pd.qcut(x, 5, labels=[1, 2, 3, 4, 5])
    )
    
    df_copy['qui_dlev'] = df_copy.groupby('year_month')['dlev'].transform(
        lambda x: pd.qcut(x, 5, labels=[1, 2, 3, 4, 5])
    )

    df_copy['qui_intan_at'] = df_copy.groupby('year_month')['intan_epk_at'].transform(
        lambda x: pd.qcut(x, 5, labels=[1, 2, 3, 4, 5])
    )
    
    df_copy['qui_debt_at'] = df_copy.groupby('year_month')['debt_at_adj'].transform(
    lambda x: pd.qcut(x, 5, labels=[1, 2, 3, 4, 5])
    )
    
    df_copy['qui_lev'] = df_copy.groupby('year_month')['lev'].transform(
    lambda x: pd.qcut(x, 5, labels=[1, 2, 3, 4, 5])
    )

    df_copy['ter_debt_at'] = df_copy.groupby('year_month')['debt_at_adj'].transform(
        lambda x: pd.qcut(x, 3, labels=[1, 2, 3])
    )

    df_copy['qua_debt_at'] = df_copy.groupby('year_month')['debt_at_adj'].transform(
        lambda x: pd.qcut(x, 4, labels=[1, 2, 3, 4])
    )

    df_copy['ter_intan_at'] = df_copy.groupby('year_month')['intan_epk_at'].transform(
        lambda x: pd.qcut(x, 3, labels=[1, 2, 3])
    )
    return df_copy





