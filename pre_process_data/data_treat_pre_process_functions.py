import pandas as pd
import sys
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

def custom_fill(group):
    group['atq'] = group['atq'].interpolate()
    group['capxy'] = group['capxy'].interpolate()
    group['cash_at'] = group['cash_at'].interpolate()
    group['debt_at'] = group['debt_at'].interpolate()
    group['org_cap_comp'] = group['org_cap_comp'].interpolate()
    group['ppentq'] = group['ppentq'].interpolate()
    group['state'] = group['state'].ffill()
    return group

def merge_comp_intan_epk(df, intan):
    
    ###################################################################################
    # Merging original Compustat with the dataframe with the intangible capital measure
    ###################################################################################
    compustat_sel = df[['datadate', 'rdq', 'GVKEY', 'sic', 
                            'atq', 'dlcq', 'dlttq', 'ceqq',
                            'state', 'niq', 'ltq', 'ppentq']]

    df_intan_sel = intan[['datadate', 'gvkey',  'k_int_t100']]

    df_intan_sel_pre_merge = (df_intan_sel
        .assign(date = pd.to_datetime(df_intan_sel["datadate"], format = '%Y-%m-%d', errors = "coerce")) #non-conforming entries will be coerced to "Not a Time - NaT"
        .assign(year_quarter = lambda x: x['date'].dt.to_period('Q'))
        .rename(columns = {'k_int_t100': 'intan_epk',
                        'gvkey': 'GVKEY'})
    )

    # df_intan_sel.head(50)
    df_intan_sel_pre_merge.head(50)

    df_intan_sel_pre_merge = df_intan_sel_pre_merge.drop(columns = ['datadate', 'date'])

    compustat_sel_pre_merge = (compustat_sel
        .assign(date = pd.to_datetime(compustat_sel["datadate"], format = '%Y-%m-%d', errors = "coerce")) #non-conforming entries will be coerced to "Not a Time - NaT"
        .assign(year_quarter = lambda x: x['date'].dt.to_period('Q'))
        .drop(columns = ['datadate'])
        .query('~(sic >= 6000 and sic < 7000) and ~(sic >= 4900 and sic < 5000) and ceqq > 0 and ltq >= 0')
    )
    # compustat_sel_pre_merge.head(50)

    # compustat_sel_pre_merge['GVKEY'].nunique()
    # compustat_sel['GVKEY'].nunique()
    # compustat_sel_pre_merge = compustat_sel_pre_merge[~compustat_sel_pre_merge.duplicated('GVKEY_year_quarter', keep='first')]
    #compustat_sel_pre_merge.shape

    compust = (pd.merge(compustat_sel_pre_merge, df_intan_sel_pre_merge, how = 'left', on = ['year_quarter', 'GVKEY']))

    # Check for duplicates in terms of year_quarter and GVKEY
    # duplicates = compust[compust.duplicated(subset=['year_quarter', 'GVKEY'], keep=False)]
    # print(duplicates)
    compust_no_dup = compust.drop_duplicates(subset=['year_quarter', 'GVKEY'], keep='first')

    # compustat_sel.shape
    # compustat_sel_pre_merge.shape
    # compust.shape
    # compust_no_na = compust.dropna(subset = ['atq_intan'])
    # compust.head(50)

    ###################################################################
    # Changing the date on Compustat to merge with CRSP (release date)
    ###################################################################

    compust = compust_no_dup.drop(columns = ['year_quarter'])
    # compust_pre_merge[compust_pre_merge['GVKEY'] == 180562][['GVKEY', 'year_quarter', 'year_month', 'intan_epk']].head(50)

    compust_pre_merge = (compust
        .assign(rdq = pd.to_datetime(compust["rdq"], format = '%Y-%m-%d', errors = "coerce")) #non-conforming entries will be coerced to "Not a Time - NaT"
        .assign(year_month = lambda x: x['date'].dt.to_period('M')) #merging according to the fiscal quarter end, instead of the release date
        .assign(lev = lambda x: x['ltq'] / x['atq'])
        .sort_values(by = ['GVKEY', 'date'])
        .assign(dlev = lambda x: x.groupby('GVKEY')['lev'].diff())
    )
    return compust_pre_merge

    # compust_pre_merge.columns
    # compust_pre_merge[compust_pre_merge['GVKEY'] == 180562][['GVKEY', 'date', 'year_quarter', 'rdq', 'year_month', 'intan_epk']].head(50)

    # compust.shape
    # compust_pre_merge.head(50)
    # compust_pre_merge.shape

    # compust_pre_merge.to_feather('data/feather/compust_pre_merge.feather')

def merge_crsp_comp(df, crsp, ff):
    df_copy = df.copy()
    # mkt_rf = ff[['date', 'Mkt-RF']]
    mkt_rf = (ff
            .assign(date = pd.to_datetime(ff['date'], format='%Y%m'))
            .assign(year_month = lambda x: x['date'].dt.to_period('M'))
            .rename(columns = {'Mkt-RF': 'mkt_rf', 'SMB': 'smb', 'HML': 'hml', 'RF': 'rf'})
            .drop(columns = ['date']))

    link_permno_gvkey = pd.read_csv('data/csv/link_permno_gvkey.csv')
    link_permno_gvkey = link_permno_gvkey.loc[:,['GVKEY', 'LPERMNO']]
    link_permno_gvkey_unique = link_permno_gvkey.drop_duplicates()
    link_permno_gvkey_renamed = link_permno_gvkey_unique.rename(columns={'LPERMNO': 'PERMNO'})

    ####################################################################
    # Adapting Compustat (generated for the merge with daily CRSP data) 
    # to merge with monthly CRSP data
    ####################################################################
    # compust_pre_merge.columns
    compust_monthly = (df_copy
                    #.assign(GVKEY_year_month = lambda x: x['GVKEY'].astype(str) + '_' + x['year'].astype(str) + '_' + x['month'].astype(str))
                    #.drop(columns = ['year_month', 'GVKEY'])
                    .rename(columns = {'date': 'date_compustat'}))
    #compust_monthly.head(50)
    ####################
    # Merging CRSP data
    ####################
    crsp_sel = crsp[['PERMNO', 'date', 'RET', 'PRC', 'SHROUT', 'EXCHCD', 'SHRCD']]
    # crsp_sel.head(50)
    crsp_clean = (crsp_sel
                    .assign(PERMNO = lambda x: x['PERMNO'].astype(int))
                    #.assign(date = lambda x: x['date'].astype(int))
                    .assign(date = pd.to_datetime(crsp["date"], format = '%Y%m%d', errors = "coerce")) #non-conforming entries will be coerced to "Not a Time - NaT"
                    .assign(year_month = lambda x: x['date'].dt.to_period('M'))
                    #.assign(year = lambda x: x['date'].dt.year.astype('Int64'))
                    #.query('1997 <= year <= 2005')
                    )
    # crsp_clean.shape
    crsp_link = (pd.merge(crsp_clean, link_permno_gvkey_renamed, how = 'left', on = 'PERMNO'))
    # crsp_link.shape

    # crsp_d_df_link = (crsp_link
    #                   .assign(GVKEY = lambda x: x['GVKEY'].astype('Int64')))

    # sample_crsp_d_df_link = crsp_d_df_link.sample(frac = 0.001, random_state = 42)
    # crsp_link.head(50)

    crsp_pre_merge = (crsp_link
                    .assign(GVKEY = lambda x: x['GVKEY'].astype('Int64'))
                    )


    crsp_pre_merge = (pd.merge(crsp_pre_merge, mkt_rf, how = 'left', on = 'year_month'))
    # crsp_pre_merge.shape

    # duplicates = compust[compust.duplicated(subset=['year_quarter', 'GVKEY'], keep=False)]
    # print(duplicates)
    crsp_pre_merge_no_dup = crsp_pre_merge.drop_duplicates(subset=['year_month', 'GVKEY'], keep='first')
    # crsp_pre_merge_no_dup.shape
    # crsp_pre_merge.shape
    # crsp_pre_merge_no_dup.shape

    crsp_merge = (crsp_pre_merge_no_dup
            .rename(columns = {'date': 'date_ret'})
            .drop(columns = ['PERMNO']))
    # crsp_merge.head(50)
    # compust_monthly.head(50)

    ccm_monthly = (pd.merge(crsp_merge, compust_monthly, how = 'left', on = ['year_month', 'GVKEY']))
    # ccm_monthly.shape
    ccm_monthly_no_dup = ccm_monthly.drop_duplicates(subset=['year_month', 'GVKEY'], keep='first')

    # ccm_monthly_no_dup.shape
        
    # ccm_monthly.columns
    # ccm_monthly.shape
    # ccm_monthly_no_dup.head(50)

    # ccm_monthly_no_dup.set_index(['GVKEY', 'year_month'], inplace=True)
    columns_fill = ['atq', 'ceqq', 'dlttq', 'dlcq', 'niq', 'sic', 'state', 'ppentq', 'ltq', 'intan_epk', 'lev', 'dlev']

    # Fill forward the missing values within each group
    ccm_monthly_filled = ccm_monthly_no_dup.copy()
    ccm_monthly_filled[columns_fill] = ccm_monthly_filled.groupby('GVKEY')[columns_fill].transform(lambda x: x.ffill())
    
    return ccm_monthly_filled


def merge_pt(df, pt):
    df_copy = df.copy()
    
    df_copy = (df_copy
      .assign(year = df_copy['date_ret'].dt.year)
      .assign(month = df_copy['date_ret'].dt.month))

    ##############################################################
    # date_compustat has many missing values. 
    # To merge with Peters and Taylor intangible capital measure,
    # I create a new column date_compustat_filled that assumes that 
    # financial statements released in the first quarter of the year
    # refer to fiscal year t-1. In that way, I fill several missing
    # values in date_compustat.
    ##############################################################

    df_copy['date_compustat_filled'] = df_copy.apply(
        lambda row: f"{row['year']-1}-12-31" if pd.isna(row['date_compustat']) and row['month'] in [1, 2, 3] else row['date_compustat'],
        axis=1
    )

    # df.columns
    # df[['date_compustat', 'date_compustat_filled']].head(50)

    # Convert the new column to datetime format
    df_copy['date_compustat_filled'] = pd.to_datetime(df_copy['date_compustat_filled'])
    # df[['GVKEY', 'date_ret', 'date_compustat', 'date_compustat_filled']].head(50)
    # df_test = df.dropna(subset = ['date_compustat_filled'])
    pt_intan_sel = pt[['gvkey', 'datadate', 'K_int']]
    # pt_intan_sel.head(50)
    pt_intan_sel['date_compustat_filled'] = pd.to_datetime(pt_intan_sel['datadate'], format='%Y%m%d', errors='coerce')
    pt_intan_sel = (pt_intan_sel
                    .drop(columns=['datadate'])
                    .rename(columns={'K_int': 'intan_pt', 'gvkey': 'GVKEY'}))
    # pt_intan_sel.head(50)

    df_intan = pd.merge(df_copy, pt_intan_sel, how = 'left', on = ['GVKEY', 'date_compustat_filled'])
    # df.shape
    # df_intan.shape

    df_intan['intan_pt_filled'] = df_intan.groupby('GVKEY')['intan_pt'].transform(lambda x: x.ffill())
    df_intan_pt = (df_intan
                .assign(year = df_intan['date_ret'].dt.year)
                #.query('year <= 2017')
                .drop(columns = ['year', 'month']))

    # df_intan[['GVKEY', 'date_ret', 'date_compustat_filled', 'intan_pt', 'intan_pt_filled']][df_intan['GVKEY'] == 11217].tail(50)  
    return df_intan_pt

def prep_fm(df, betas):
    df_copy = df.copy()
    
    df_copy = (df_copy
        .assign(year = df_copy['date_ret'].dt.year)
        .query('year >= 1975 and (EXCHCD == 1 | EXCHCD == 2 | EXCHCD == 3) and (SHRCD == 10 | SHRCD == 11)')) #and ltq >= 0
    # df.shape
    df_copy = (pd.merge(df_copy, betas, how = 'left', on = ['GVKEY', 'year_month']))
    df_copy['debt_at'] = (df_copy['dlttq'] + df_copy['dlcq']) / df_copy['atq']
    # df_copy['lev'] = df_copy['ltq'] / df_copy['atq']
    df_copy['intan_pt_at'] = df_copy['intan_pt_filled'] / df_copy['atq']
    df_copy['intan_epk_at'] = df_copy['intan_epk'] / df_copy['atq']
    df_copy['roe'] = df_copy['niq'] / df_copy['ceqq']
    df_copy['roa'] = df_copy['niq'] / df_copy['atq']

    # Change book-to-market ratio
    df_copy = (df_copy
        #.assign(ceqq_lag1 = df.groupby('GVKEY')['ceqq'].shift(1))
        .assign(me = (np.abs(df_copy['PRC'])*df_copy['SHROUT'])/1000000))# convert to billions
    df_copy = (df_copy
        .assign(bm = df_copy['ceqq'] / (df_copy['me']*1000)) #ceqq is in millions and me is in billions
        .assign(ln_ceqq = np.log(df_copy['ceqq']))
        .assign(ln_me = np.log(df_copy['me']))      
        .assign(ln_at = np.log(df_copy['atq']))
        .assign(RET = pd.to_numeric(df_copy['RET'], errors='coerce'))
        .assign(year_month = df_copy['date_ret'].dt.to_period('M'))
        )

    # Removing outliers based on book-to-market ratio
    lower_bound = df_copy['bm'].quantile(0.01)
    upper_bound = df_copy['bm'].quantile(0.99)
    df_copy = df_copy[(df_copy['bm'] >= lower_bound) & (df_copy['bm'] <= upper_bound)]

    # df[['GVKEY', 'year_month', 'ceqq', 'ceqq_lag1', 'PRC', 'SHROUT', 'bm']].tail(50)

    df_copy['year_month'] = df_copy['year_month'].dt.to_timestamp()
    df_copy.set_index(['GVKEY', 'year_month'], inplace=True)

    df_copy['terc_lev'] = df_copy.groupby('year_month')['lev'].transform(
        lambda x: pd.qcut(x, 3, labels=['Low', 'Medium', 'High'])
    )

    
    df_new = (df_copy
        .assign(RET = pd.to_numeric(df_copy['RET'],  errors = "coerce"))
        .assign(ret_aux = lambda x: 1 + x['RET'])
        .assign(one_year_cum_return = lambda x: x.groupby('GVKEY')['ret_aux'].rolling(window=12, min_periods=12).apply(np.prod, raw=True).reset_index(level=0, drop=True) - 1)
        #.assign(one_year_cum_return = lambda x: x.groupby('GVKEY')['cum_return'].shift(-11))  # Align with current date
        .assign(ret_aux_lead1 = lambda x: x.groupby('GVKEY')['ret_aux'].shift(-1))
        .assign(ret_aux_lead2 = lambda x: x.groupby('GVKEY')['ret_aux'].shift(-2))
        .assign(ret_2mo = lambda x: (x['ret_aux']*x['ret_aux_lead1']) - 1)
        .assign(ret_2mo_lead1 = lambda x: x.groupby('GVKEY')['ret_2mo'].shift(-1))
        .assign(ret_3mo = lambda x: (x['ret_aux']*x['ret_aux_lead1']*x['ret_aux_lead2']) - 1)
        .assign(ret_3mo_lead1 = lambda x: x.groupby('GVKEY')['ret_3mo'].shift(-1))
        .assign(hlev = lambda x: x['terc_lev'] == 'High')
        .assign(llev = lambda x: x['terc_lev'] == 'Low')
        # .assign(hint = lambda x: x['qua_intan'] == 4)
        # .assign(lint = lambda x: x['qua_intan'] == 1)                
        .drop(columns=['ret_aux', 'ret_aux_lead1', 'ret_aux_lead2'])       
        )

    # df_new[['RET', 'one_year_cum_return']].head(50)
    #df_new[['RET', 'ret_aux', 'ret_aux_lead1', 'ret_aux_lead2', 'ret_3mo', 'ret_3mo_lead1']].head(50)
    # df_new[['hint', 'quart_intan']][df_new['quart_intan'] == 4].tail(50)
    # df_int = df_new.query('hint == True')

    df_new['RET_lead1'] = df_new.groupby('GVKEY')['RET'].shift(-1)
    df_new['debt_at_lag1'] = df_new.groupby('GVKEY')['debt_at'].shift(1)
    df_new['d_debt_at'] = df_new['debt_at'] - df_new['debt_at_lag1']
    df_new['d_ln_debt_at'] = np.log(df_new['debt_at']) - np.log(df_new['debt_at_lag1'])
    df_new['lev_lag1'] = df_new.groupby('GVKEY')['lev'].shift(1)
    df_new['d_lev'] = df_new['lev'] - df_new['lev_lag1']
    df_new['d_ln_lev'] = np.log(df_new['lev']) - np.log(df_new['lev_lag1'])
    df_new['roe_lag1'] = df_new.groupby('GVKEY')['roe'].shift(1)
    df_new['d_roe'] = df_new['roe'] - df_new['roe_lag1']

    #df_int['dummyXdebt_at'] = df_int['debt_at'] * df_int['lint']

    df_clean = df_new.copy()
    df_clean['ln_ceqq'] = df_clean['ln_ceqq'].replace([np.inf, -np.inf], np.nan)
    df_clean['ln_me'] = df_clean['ln_me'].replace([np.inf, -np.inf], np.nan)
    df_clean['ln_at'] = df_clean['ln_at'].replace([np.inf, -np.inf], np.nan)
    df_clean['d_debt_at'] = df_clean['d_debt_at'].replace([np.inf, -np.inf], np.nan)
    df_clean['d_ln_debt_at'] = df_clean['d_ln_debt_at'].replace([np.inf, -np.inf], np.nan)
    df_clean['d_lev'] = df_clean['d_lev'].replace([np.inf, -np.inf], np.nan)
    df_clean['d_ln_lev'] = df_clean['d_ln_lev'].replace([np.inf, -np.inf], np.nan)
    df_clean['d_roe'] = df_clean['d_roe'].replace([np.inf, -np.inf], np.nan)
    df_clean['dlev'] = df_clean['dlev'].replace([np.inf, -np.inf], np.nan)

    
    return df_clean


def calc_beta(df):
    df_copy = df.copy()
    
    df_short = (df_copy
        .assign(year = df_copy['date_ret'].dt.year)
        .query('year >= 1975'))

    df_sel = df_short[['year_month', 'GVKEY', 'RET', 'mkt_rf', 'rf']]


    # rf_monthly = (rf
    #             .assign(date_ret = pd.to_datetime(rf['DATE'], format='%Y-%m-%d', errors = "coerce"))
    #             .assign(year_month = lambda x: x['date_ret'].dt.to_period('M'))
    #             .assign(rf_annual = lambda x: pd.to_numeric(x['TB3MS'], errors = "coerce"))
    #             .assign(rf_monthly = lambda x: ((1 + x['rf_annual']/100)**(1/12)) - 1)
    #             .drop(columns= ['DATE', 'date_ret', 'TB3MS', 'rf_annual'])
    #             )
    #rf_monthly['rf_annual'].dtype
    #rf_monthly[rf_monthly['year_month'] == '2023-12'].head(50)
    #rf_monthly.head(50)
    # df.columns
    # df['mkt_rf'].isna().sum()

    # df_beta = (pd.merge(df_sel, rf_monthly, how = 'left', on = 'year_month'))
    # rf_monthly.columns
    # df_sel.columns
    # df_beta.head(50)
    # df_sel.shape
    df_reg = (df_sel
            .assign(RET = pd.to_numeric(df_sel['RET'], errors = 'coerce'))
            .assign(ret_rf = lambda x: x['RET'] - x['rf'])
            .assign(mkt_rf = df_sel['mkt_rf']/100)
            #.assign(mkt_rf = pd.to_numeric(df_beta['mkt_rf'], errors = 'coerce'))
            .drop(columns = ['RET', 'rf'])   
    )
    # df_reg.head(50)
    ########################################################
    # Run a rolling window OLS regression to calculate beta
    ########################################################

    df_reg = df_reg.sort_values(by=['GVKEY', 'year_month'])

    # Apply the rolling regression with a 2-year window (24 months)
    rolling_results = rolling_regression(df_reg, 24)

    # print(rolling_results)
    # rolling_results.shape
    # rolling_results_no_na = rolling_results.dropna(subset = ['mkt_rf'])
    # rolling_results_no_na.shape
    betas = rolling_results[['year_month', 'GVKEY', 'mkt_rf']]
    betas = betas.rename(columns = {'mkt_rf': 'beta'})
    # betas.head(50)

    # Merge the betas with the original dataframe
    df_reg_beta = df_reg.merge(betas, on=['year_month', 'GVKEY'], how='left')
    # df_reg_beta.shape
    # df_reg.shape
    df_reg_beta = df_reg_beta[['year_month', 'GVKEY', 'beta']]
    return df_reg_beta
    
    
def rolling_regression(df, window_length):
    #df['year_month'] = pd.to_datetime(df['year_month'].astype(str), format='%Y-%m')
    results = []

    for key, group in df.groupby('GVKEY'):
        group = group.dropna(subset=['mkt_rf', 'ret_rf'])
        if len(group) < window_length:
            # Skip groups with fewer data points than the rolling window size
            continue

        group = group.set_index('year_month')
        group = group.sort_index()
        
        # Prepare the dependent and independent variables
        y = group['ret_rf']
        X = sm.add_constant(group['mkt_rf'])

        # Apply RollingOLS
        rolling_model = RollingOLS(y, X, window=window_length)
        rolling_params = rolling_model.fit().params

        # Add GVKEY and reset index
        rolling_params = rolling_params.reset_index()
        rolling_params['GVKEY'] = key
        results.append(rolling_params)
    
    results_df = pd.concat(results)
    return results_df