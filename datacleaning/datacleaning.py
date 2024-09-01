import pandas as pd
import sys
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from invest_strat import announce_execution
import pickle

def csv_to_pickle():
    print("Converting CSV files to pickle files")
    df = pd.read_csv("../data/csv/comp_fundq.csv") # origianl Compustat
    compustat_sel = df[['datadate', 'rdq', 'GVKEY', 'sic', 
                            'atq', 'dlcq', 'dlttq', 'ceqq',
                            'state', 'niq', 'ltq', 'ppentq']]
    compustat_sel.to_pickle("../data/pickle/comp_fundq.pkl")
    print("Converted Compustat")
    crsp = pd.read_csv('../data/csv/crsp_full.csv') # original CRSP
    crsp_sel = crsp[['PERMNO', 'date', 'RET', 'PRC', 'SHROUT', 'EXCHCD', 'SHRCD']]
    crsp_sel.to_pickle("../data/pickle/crsp_full.pkl")
    print("Converted CRSP")
    ff = pd.read_csv('../data/csv/F-F_Research_Data_Factors.CSV') # from Kenneth French's website
    ff.to_pickle("../data/pickle/F-F_Research_Data_Factors.pkl")
    print("Converted Fama-French factors")
    pt = pd.read_csv('../data/csv/peterstaylor.csv')
    pt.to_pickle("../data/pickle/peterstaylor.pkl")
    print("Converted Peters and Taylor intangible capital data")
    intan = pd.read_csv('../data/csv/int_xsec.csv') # from
    intan.to_pickle("../data/pickle/int_xsec.pkl")
    print("Converted intangible capital data")
    kkr = pd.read_csv('../data/csv/kkr.csv') # from
    kkr.to_pickle("../data/pickle/kkr.pkl")
    print("Converted knowledge capital risk data")


@announce_execution
def load_fm(redo = False, origin = "pickle"):
    if redo:
        if origin == 'csv':
            print("Reading from CSV files")
            print("Reading intangible capital data")
            intan = pd.read_csv('../data/csv/int_xsec.csv') # from https://github.com/edwardtkim/intangiblevalue/tree/main/output
            print("Reading original Compustat and CRSP data")
            df = pd.read_csv("../data/csv/comp_fundq.csv") # origianl Compustat
            print("Reading CRSP data")
            crsp = pd.read_csv('../data/csv/crsp_full.csv') # original CRSP
            print("Reading Fama-French factors")
            ff = pd.read_csv('../data/csv/F-F_Research_Data_Factors.CSV') # from Kenneth French's website
            print("Reading Peters and Taylor intangible capital data")
            pt = pd.read_csv('../data/csv/peterstaylor.csv')
        elif origin == 'pickle' or origin == 'pkl':
            print("Reading from pickle files")
            print("Reading intangible capital data")
            intan = pd.read_pickle('../data/pickle/int_xsec.pkl')
            print("Reading original Compustat and CRSP data")
            df = pd.read_pickle("../data/pickle/comp_fundq.pkl")
            print("Reading CRSP data")
            crsp = pd.read_pickle('../data/pickle/crsp_full.pkl')
            print("Reading Fama-French factors")
            # ff = pd.read_pickle('../data/pickle/F-F_Research_Data_Factors.pkl')
            ff = pd.read_pickle('../data/pickle/F-F_Research_Data_5_Factors_2x3.pkl')
            mom = pd.read_pickle('../data/pickle/F-F_Research_Data_Factors.pkl')
            print("Reading Peters and Taylor intangible capital data")
            pt = pd.read_pickle('../data/pickle/peterstaylor.pkl')
            print("Reading knowledge capital risk data")
            kkr = pd.read_pickle('../data/pickle/kkr.pkl')
        else:
            raise ValueError("Invalid origin. Please use 'csv' or 'pickle' as origin.")

        comp_intan = merge_comp_intan_epk(df, intan)
        cc = merge_crsp_comp(comp_intan, crsp, ff, mom) # also merges with Fama-French factors
        cc_pt = merge_pt(cc, pt)
        betas = calc_beta(cc_pt)
        df_fm = prep_fm(cc_pt, betas, kkr)
        df_fm = df_fm.reset_index()
        df_fm = compute_default_probabilities(df_fm, progress_interval=1000)
        # df_fm.set_index(['GVKEY', 'year_month'], inplace=True)
        
        df_fm.to_feather('../data/feather/df_fm.feather')
    else:
        df_fm = pd.read_feather('../data/feather/df_fm.feather') #from prep_fm.py (folder porfolio)
        
        # port_ff25 = pd.read_pickle("../data/pickle/25_Portfolios_5x5.pkl")
        port_ff6 = pd.read_pickle("../data/pickle/6_Portfolios_2x3.pkl")

        # # port_ff25 = (port_ff25
        # #              .assign(date = pd.to_datetime(port_ff25['date'], format='%Y%m'))
        # #              .assign(year_month = lambda x: x['date'].dt.to_period('M'))
        # #              .drop(columns = ['date'])
        # # )
        
        port_ff6 = (port_ff6
                    .assign(date = pd.to_datetime(port_ff6['date'], format='%Y%m'))
                    .assign(year_month = lambda x: x['date'].dt.to_period('M'))
                    .assign(year_month = lambda x: x['year_month'].dt.to_timestamp())
                    .drop(columns = ['date'])
        )
    
        # Pivot this by melting the dataframe
        port_ff6_pivot = (port_ff6
                          .melt(id_vars = ['year_month'], var_name = 'port', value_name = 'ret')
                          .assign(port = lambda x: x['port'].str.replace(' ', ''))
        )
        
        # Replace string identifiers with numbers from 1 to 6
        port_ff6_pivot['port_id'] = pd.factorize(port_ff6_pivot['port'])[0] + 1
        port_ff6_pivot = port_ff6_pivot.drop(columns = ['port'])
        
    return df_fm, port_ff6_pivot

@announce_execution
def merge_comp_intan_epk(df, intan):
    
    ###################################################################################
    # Merging original Compustat with the dataframe with the intangible capital measure
    ###################################################################################
    compustat_sel = df[['datadate', 'rdq', 'GVKEY', 'sic', 
                            'atq', 'dlcq', 'dlttq', 'ceqq',
                            'state', 'niq', 'ltq', 'ppentq']]

    df_intan_sel = intan[['datadate', 'gvkey',  'k_int_t100', 'ok']]

    df_intan_sel_pre_merge = (df_intan_sel
        .assign(date = pd.to_datetime(df_intan_sel["datadate"], format = '%Y-%m-%d', errors = "coerce")) #non-conforming entries will be coerced to "Not a Time - NaT"
        .assign(year_quarter = lambda x: x['date'].dt.to_period('Q'))
        .rename(columns = {'k_int_t100': 'intan_epk',
                        'gvkey': 'GVKEY'})
    )

    df_intan_sel_pre_merge = df_intan_sel_pre_merge.drop(columns = ['datadate', 'date'])

    compustat_sel_pre_merge = (compustat_sel
        .assign(date = pd.to_datetime(compustat_sel["datadate"], format = '%Y-%m-%d', errors = "coerce")) #non-conforming entries will be coerced to "Not a Time - NaT"
        .assign(year_quarter = lambda x: x['date'].dt.to_period('Q'))
        .drop(columns = ['datadate'])
        .query('~(sic >= 6000 and sic < 7000) and ~(sic >= 4900 and sic < 5000) and ceqq > 0 and ltq >= 0')
    )


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

@announce_execution
def merge_crsp_comp(df, crsp, ff, mom):
    df_copy = df.copy()
    # mkt_rf = ff[['date', 'Mkt-RF']]
    mkt_rf = (ff
            .assign(date = pd.to_datetime(ff['date'], format='%Y%m'))
            .assign(year_month = lambda x: x['date'].dt.to_period('M'))
            .rename(columns = {'Mkt-RF': 'mkt_rf', 'SMB': 'smb', 'HML': 'hml', 'RF': 'rf', 'CMA': 'cma', 'RMW': 'rmw'})
            .drop(columns = ['date']))
    
    mom = (mom
            .assign(date = pd.to_datetime(mom['date'], format='%Y%m'))
            .assign(year_month = lambda x: x['date'].dt.to_period('M'))
            .rename(columns = {'Mom': 'mom'})
            .drop(columns = ['date']))

    link_permno_gvkey = pd.read_csv('../data/csv/link_permno_gvkey.csv')
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
    crsp_pre_merge = (pd.merge(crsp_pre_merge, mom, how = 'left', on = 'year_month'))
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

    # ccm_monthly_no_dup.set_index(['GVKEY', 'year_month'], inplace=True)
    columns_fill = ['atq', 'ceqq', 'dlttq', 'dlcq', 'niq', 'sic', 'state', 'ppentq', 'ltq', 'lev', 'dlev']
    col_intan_fill = ['intan_epk', 'ok']

    # Fill forward the missing values within each group
    ccm_monthly_filled = ccm_monthly_no_dup.copy()
    ccm_monthly_filled[columns_fill] = ccm_monthly_filled.groupby('GVKEY')[columns_fill].transform(lambda x: x.ffill(limit = 2))
    ccm_monthly_filled[col_intan_fill] = ccm_monthly_filled.groupby('GVKEY')[col_intan_fill].transform(lambda x: x.ffill(limit = 11))

    return ccm_monthly_filled

@announce_execution
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

@announce_execution
def prep_fm(df, betas, kkr):
    df_copy = df.copy()
    df_copy = ff_indust(df_copy)
    
    df_copy = (df_copy
        .assign(year = df_copy['date_ret'].dt.year)
        .query('year >= 1975 and (EXCHCD == 1 | EXCHCD == 2 | EXCHCD == 3) and (SHRCD == 10 | SHRCD == 11)')) #already filtered ceqq > 0 and ltq >= 0 in merge_comp_intan_epk
    # df.shape
    df_copy = (pd.merge(df_copy, betas, how = 'left', on = ['GVKEY', 'year_month']))
    kkr = (kkr
           .assign(GVKEY = kkr['gvkey'].astype('Int64'))
           )
    kkr = kkr[['GVKEY', 'year', 'KKR']]
    df_copy = (pd.merge(df_copy, kkr, how = 'left', on = ['GVKEY', 'year']))
    df_copy = df_copy.sort_values(by=['GVKEY', 'year_month'])
    col_intan_fill = ['KKR']
    df_copy[col_intan_fill] = df_copy.groupby('GVKEY')[col_intan_fill].transform(lambda x: x.ffill(limit = 11))
    df_copy['debt_at'] = (df_copy['dlttq'] + df_copy['dlcq']) / df_copy['atq']
    # df_copy['lev'] = df_copy['ltq'] / df_copy['atq']
    df_copy['intan_pt_at'] = df_copy['intan_pt_filled'] / df_copy['atq']
    df_copy['intan_epk_at'] = df_copy['intan_epk'] / df_copy['atq']
    df_copy['roe'] = df_copy['niq'] / df_copy['ceqq']
    df_copy['roa'] = df_copy['niq'] / df_copy['atq']
    df_copy['ceqq_lag3'] = df_copy.groupby('GVKEY')['ceqq'].shift(3) #lagging book value of equity by one quarter to compute bm


    # Change book-to-market ratio
    df_copy = (df_copy
        #.assign(ceqq_lag1 = df.groupby('GVKEY')['ceqq'].shift(1))
        .assign(me = (np.abs(df_copy['PRC'])*df_copy['SHROUT'])/1000000))# convert to billions
    df_copy = (df_copy
        .assign(bm = df_copy['ceqq_lag3'] / (df_copy['me']*1000)) #ceqq is in millions and me is in billions
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
        .assign(ret_lead1 = lambda x: x.groupby('GVKEY')['RET'].shift(-1))
        .assign(ret_lead2 = lambda x: x.groupby('GVKEY')['RET'].shift(-2))
        .assign(ret_lead3 = lambda x: x.groupby('GVKEY')['RET'].shift(-3))
        # .assign(hint = lambda x: x['qua_intan'] == 4)
        # .assign(lint = lambda x: x['qua_intan'] == 1)                
        .drop(columns=['ret_aux', 'ret_aux_lead1', 'ret_aux_lead2'])       
        )
    
    # df_new[['RET', 'one_year_cum_return']].head(50)
    #df_new[['RET', 'ret_aux', 'ret_aux_lead1', 'ret_aux_lead2', 'ret_3mo', 'ret_3mo_lead1']].head(50)
    # df_new[['hint', 'quart_intan']][df_new['quart_intan'] == 4].tail(50)
    # df_int = df_new.query('hint == True')

    df_new['ret_lag1'] = df_new.groupby('GVKEY')['RET'].shift(1)
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

# def merton_model(me, sigma_e, debt, rf, time_horizon):
#     """
#     Implement the Merton (1974) model to compute default probability.
#     """
#     me = max(me, 1e-6)
#     sigma_e = max(sigma_e, 1e-6)
#     debt = max(debt, 1e-6)

#     asset_value = me + debt
#     asset_volatility = sigma_e * me / asset_value

#     for i in range(100):
#         d1 = (np.log(asset_value / debt) + (rf + 0.5 * asset_volatility**2) * time_horizon) / (asset_volatility * np.sqrt(time_horizon))
#         d2 = d1 - asset_volatility * np.sqrt(time_horizon)
        
#         new_asset_value = me + debt * np.exp(-rf * time_horizon) * norm.cdf(d2)
#         new_asset_volatility = sigma_e * me / (new_asset_value * norm.cdf(d1))
        
#         if abs(new_asset_value - asset_value) < 1e-6 and abs(new_asset_volatility - asset_volatility) < 1e-6:
#             break
        
#         asset_value = new_asset_value
#         asset_volatility = new_asset_volatility

#     distance_to_default = d2
#     default_probability = norm.cdf(-distance_to_default)

#     return default_probability, distance_to_default

def merton_model(me, sigma_e, debt, rf, time_horizon, max_iterations=200, tolerance=1e-4):
    """
    Implement the Merton (1974) model to compute default probability with improved convergence.
    """
    # Ensure positive values
    me = me*1000 # convert to millions
    me = max(me, 1e-6)
    sigma_e = max(sigma_e, 1e-6)
    debt = max(debt, 1e-6)

    # Improved initial guess
    asset_value = me + debt
    asset_volatility = sigma_e * (me / asset_value)

    for i in range(max_iterations):
        d1 = (np.log(asset_value / debt) + (rf + 0.5 * asset_volatility**2) * time_horizon) / (asset_volatility * np.sqrt(time_horizon))
        d2 = d1 - (asset_volatility * np.sqrt(time_horizon))
        
        new_asset_value = (me + debt * np.exp(-rf * time_horizon) * norm.cdf(d2))/norm.cdf(d1)
        new_asset_volatility = sigma_e * me / (new_asset_value * norm.cdf(d1))
        
        # Prevent extreme values
        # new_asset_value = np.clip(new_asset_value, me, me + debt * 10)
        # new_asset_volatility = np.clip(new_asset_volatility, sigma_e * 0.1, sigma_e * 10)
        
        # Check for convergence using relative tolerance
        if (abs(new_asset_value - asset_value) < tolerance * asset_value and 
            abs(new_asset_volatility - asset_volatility) < tolerance * asset_volatility):
            break
        
        asset_value = new_asset_value
        asset_volatility = new_asset_volatility
    
    # Compute final values even if max iterations reached
    d2 = (np.log(asset_value / debt) + (rf - 0.5 * asset_volatility**2) * time_horizon) / (asset_volatility * np.sqrt(time_horizon))
    distance_to_default = d2
    
    default_probability = norm.cdf(-distance_to_default)

    # Return results along with convergence status
    converged = i < max_iterations - 1
    return default_probability, distance_to_default, converged

def compute_equity_volatility(group, window=12):
    group['sigma_e'] = group['RET'].rolling(window=window).std() * np.sqrt(12)
    # group['sigma_e'] = group['RET'].std() * np.sqrt(12)
    return group

def apply_merton(row):
    if pd.isna(row['sigma_e']):
        return pd.Series({'default_probability': np.nan, 'distance_to_default': np.nan, 'converged': False})
    default_probability, distance_to_default, converged = merton_model(
        row['me'],
        row['sigma_e'],
        row['ltq'],
        row['rf'],
        # row['atq'],
        time_horizon=40.56
    )
    return pd.Series({'default_probability': default_probability, 'distance_to_default': distance_to_default, 'converged': converged})

def compute_default_probabilities(df, progress_interval=1000):
    df_copy = df.copy()
    # df_copy = df_copy.reset_index()
    df_copy = df_copy.sort_values(['GVKEY', 'year_month'])

    # Compute equity volatility for each firm
    print("Computing equity volatility...")
    grouped = df_copy.groupby('GVKEY')
    df_copy = grouped.apply(compute_equity_volatility)

    print("Applying Merton model...")
    total_rows = len(df_copy)
    results = []
    non_converged = 0
    
    for count, (_, row) in enumerate(df_copy.iterrows(), 1):
        result = apply_merton(row)
        results.append(result)
        
        if not result['converged']:
            non_converged += 1
        
        if count % progress_interval == 0:
            conv_rate = (count - non_converged) / count * 100
            print(f"Processed {count} / {total_rows} observations ({count / total_rows * 100:.2f}%)")
            print(f"Convergence rate: {conv_rate:.2f}%")
    
    df_copy[['default_probability', 'distance_to_default', 'converged']] = pd.DataFrame(results, index=df_copy.index)

    print("Computation complete!")
    print(f"Overall convergence rate: {(total_rows - non_converged) / total_rows * 100:.2f}%")
    return df_copy

# def merton_model_improved(me, sigma_e, debt, rf, time_horizon, max_iterations=100, tolerance=1e-6):
#     # print(f"Input parameters: me={me}, sigma_e={sigma_e}, debt={debt}, rf={rf}, time_horizon={time_horizon}")
    
#     # Ensure positive values
#     me = max(me, 1e-6)
#     sigma_e = max(sigma_e, 1e-6)
#     debt = max(debt, 1e-6)

#     # Initial guess
#     asset_value = me + debt
#     asset_volatility = sigma_e * me / asset_value

#     for i in range(max_iterations):
#         try:
#             d1 = (np.log(asset_value / debt) + (rf + 0.5 * asset_volatility**2) * time_horizon) / (asset_volatility * np.sqrt(time_horizon))
#             d2 = d1 - asset_volatility * np.sqrt(time_horizon)
            
#             new_asset_value = me + debt * np.exp(-rf * time_horizon) * norm.cdf(d2)
#             new_asset_volatility = sigma_e * me / (new_asset_value * norm.cdf(d1))
            
#             # Bound the new values
#             new_asset_value = np.clip(new_asset_value, me, me + debt * 10)
#             new_asset_volatility = np.clip(new_asset_volatility, sigma_e * 0.1, sigma_e * 10)
            
#             # print(f"Iteration {i+1}")
#             # print(f"Asset Value: {asset_value:.6f} -> {new_asset_value:.6f}")
#             # print(f"Asset Volatility: {asset_volatility:.6f} -> {new_asset_volatility:.6f}")
            
#             if (abs(new_asset_value - asset_value) < tolerance * asset_value and 
#                 abs(new_asset_volatility - asset_volatility) < tolerance * asset_volatility):
#                 # print("Convergence reached!")
#                 break
            
#             asset_value = new_asset_value
#             asset_volatility = new_asset_volatility

#         except Exception as e:
#             print(f"Error in iteration {i+1}: {str(e)}")
#             return np.nan, np.nan
    
#     else:
#         print(f"Warning: Maximum iterations ({max_iterations}) reached without convergence")

#     d2 = (np.log(asset_value / debt) + (rf - 0.5 * asset_volatility**2) * time_horizon) / (asset_volatility * np.sqrt(time_horizon))
#     distance_to_default = d2
#     default_probability = norm.cdf(-distance_to_default)

#     # print(f"Final results: default_probability={default_probability:.6f}, distance_to_default={distance_to_default:.6f}")
#     return default_probability, distance_to_default

# def compute_equity_volatility(group, window=12):
#     group['sigma_e'] = group['RET'].rolling(window=window).std() * np.sqrt(12)
#     return group

# def apply_merton(row):
#     if pd.isna(row['sigma_e']):
#         return pd.Series({'default_probability': np.nan, 'distance_to_default': np.nan})
#     default_probability, distance_to_default = merton_model_improved(
#         row['me'],
#         row['sigma_e'],
#         row['ltq'],
#         row['rf'],
#         time_horizon=1/12
#     )
#     return pd.Series({'default_probability': default_probability, 'distance_to_default': distance_to_default})

# def compute_default_probabilities(df, progress_interval=1000):
#     df_copy = df.copy()
#     df_copy = df_copy.reset_index()
#     df_copy = df_copy.sort_values(['GVKEY', 'year_month'])

#     # Compute equity volatility for each firm
#     print("Computing equity volatility...")
#     grouped = df_copy.groupby('GVKEY')
#     df_copy = grouped.apply(compute_equity_volatility)

#     # Apply Merton model with progress tracking
#     print("Applying Merton model...")
#     total_rows = len(df_copy)
#     results = []
    
#     for count, (_, row) in enumerate(df_copy.iterrows(), 1):
#         result = apply_merton(row)
#         results.append(result)
        
#         if count % progress_interval == 0:
#             print(f"Processed {count} / {total_rows} observations ({count / total_rows * 100:.2f}%)")
    
#     df_copy[['default_probability', 'distance_to_default']] = pd.DataFrame(results, index=df_copy.index)

#     print("Computation complete!")
#     return df_copy



    
def ff_indust(df):
    df['ff_indust'] = np.nan

# Consumer Nondurables
    df.loc[((df['sic'] >= 100) & (df['sic'] <= 999)) |
           ((df['sic'] >= 2000) & (df['sic'] <= 2399)) |
           ((df['sic'] >= 2700) & (df['sic'] <= 2749)) |
           ((df['sic'] >= 2770) & (df['sic'] <= 2799)) |
           ((df['sic'] >= 3100) & (df['sic'] <= 3199)) |
           ((df['sic'] >= 3940) & (df['sic'] <= 3989)), 'ff_indust'] = 1
    
    # Consumer Durables
    df.loc[((df['sic'] >= 2500) & (df['sic'] <= 2519)) |
           ((df['sic'] >= 2590) & (df['sic'] <= 2599)) |
           ((df['sic'] >= 3630) & (df['sic'] <= 3659)) |
           ((df['sic'] >= 3710) & (df['sic'] <= 3711)) |
           ((df['sic'] == 3714)) |
           ((df['sic'] == 3716)) |
           ((df['sic'] >= 3750) & (df['sic'] <= 3751)) |
           ((df['sic'] == 3792)) |
           ((df['sic'] >= 3900) & (df['sic'] <= 3939)) |
           ((df['sic'] >= 3990) & (df['sic'] <= 3999)), 'ff_indust'] = 2
    
    # Manufacturing
    df.loc[((df['sic'] >= 2520) & (df['sic'] <= 2589)) |
           ((df['sic'] >= 2600) & (df['sic'] <= 2699)) |
           ((df['sic'] >= 2750) & (df['sic'] <= 2769)) |
           ((df['sic'] >= 3000) & (df['sic'] <= 3099)) |
           ((df['sic'] >= 3200) & (df['sic'] <= 3569)) |
           ((df['sic'] >= 3580) & (df['sic'] <= 3629)) |
           ((df['sic'] >= 3700) & (df['sic'] <= 3709)) |
           ((df['sic'] >= 3712) & (df['sic'] <= 3713)) |
           ((df['sic'] == 3715)) |
           ((df['sic'] >= 3717) & (df['sic'] <= 3749)) |
           ((df['sic'] >= 3752) & (df['sic'] <= 3791)) |
           ((df['sic'] >= 3793) & (df['sic'] <= 3799)) |
           ((df['sic'] >= 3830) & (df['sic'] <= 3839)) |
           ((df['sic'] >= 3860) & (df['sic'] <= 3899)), 'ff_indust'] = 3
    
    # Oil, Gas, and Coal Extraction and Products
    df.loc[((df['sic'] >= 1200) & (df['sic'] <= 1399)) |
           ((df['sic'] >= 2900) & (df['sic'] <= 2999)), 'ff_indust'] = 4
    
    # Chemicals and Allied Products
    df.loc[((df['sic'] >= 2800) & (df['sic'] <= 2829)) |
           ((df['sic'] >= 2840) & (df['sic'] <= 2899)), 'ff_indust'] = 5
    
    # Business Equipment
    df.loc[((df['sic'] >= 3570) & (df['sic'] <= 3579)) |
           ((df['sic'] >= 3660) & (df['sic'] <= 3692)) |
           ((df['sic'] >= 3694) & (df['sic'] <= 3699)) |
           ((df['sic'] >= 3810) & (df['sic'] <= 3829)) |
           ((df['sic'] >= 7370) & (df['sic'] <= 7379)), 'ff_indust'] = 6
    
    # Telephone and Television Transmission
    df.loc[((df['sic'] >= 4800) & (df['sic'] <= 4899)), 'ff_indust'] = 7
    
    # Utilities
    df.loc[((df['sic'] >= 4900) & (df['sic'] <= 4949)), 'ff_indust'] = 8
    
    # Wholesale, Retail, and Some Services
    df.loc[((df['sic'] >= 5000) & (df['sic'] <= 5999)) |
           ((df['sic'] >= 7200) & (df['sic'] <= 7299)) |
           ((df['sic'] >= 7600) & (df['sic'] <= 7699)), 'ff_indust'] = 9
    
    # Healthcare, Medical Equipment, and Drugs
    df.loc[((df['sic'] >= 2830) & (df['sic'] <= 2839)) |
           ((df['sic'] == 3693)) |
           ((df['sic'] >= 3840) & (df['sic'] <= 3859)) |
           ((df['sic'] >= 8000) & (df['sic'] <= 8099)), 'ff_indust'] = 10
    
    # Finance
    df.loc[((df['sic'] >= 6000) & (df['sic'] <= 6999)), 'ff_indust'] = 11
    
    # Other
    df.loc[((df['sic'] >= 1000) & (df['sic'] <= 1399)) |
           ((df['sic'] >= 1400) & (df['sic'] <= 1999)) |
           ((df['sic'] >= 4000) & (df['sic'] <= 4799)) |
           ((df['sic'] >= 7000) & (df['sic'] <= 7199)) |
           ((df['sic'] >= 7300) & (df['sic'] <= 7399)) |
           ((df['sic'] >= 7500) & (df['sic'] <= 7599)) |
           ((df['sic'] >= 7700) & (df['sic'] <= 8999)) |
           ((df['sic'] >= 9100) & (df['sic'] <= 9999)), 'ff_indust'] = 12
    
    return df



@announce_execution
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