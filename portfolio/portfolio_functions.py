import pandas as pd
import numpy as np
from tabulate import tabulate
from linearmodels.panel.model import FamaMacBeth
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
import sys
sys.path.append('code/lev_stock/portfolio/')
sys.path.append('code/lev_stock/invest_strat/')
import utils as ut

def create_sum_stats(df):
    # df_copy = df.copy()
    df_vars = ['me', 'lev', 'dlev', 'bm', 'ln_at']
    variable_labels = {
    'lev': 'Leverage',
    'dlev': 'Leverage change',
    # 'debt_at': 'Leverage',
    # 'd_debt_at': 'Leverage change',
    'ln_ceqq': 'Log(equity)',
    'roa': 'Return on Assets',
    'beta': 'Beta',
    'bm': 'Book-to-Market Ratio',
    'me': 'Market value of equity',
    'ln_at': 'Log(assets)'
    }

    mean = df[df_vars].mean()
    std = df[df_vars].std()
    median = df[df_vars].median()

    summary_stats = pd.DataFrame({
    'Variable': df_vars,
    'Mean': mean,
    'Median': median,
    'Standard Deviation': std
    })

    # Apply formatting
    summary_stats['Mean'] = summary_stats.apply(lambda row: f"{row['Mean']:.4f}" if row['Variable'] == 'dlev' else f"{row['Mean']:.2f}", axis=1)
    summary_stats['Median'] = summary_stats.apply(lambda row: f"{row['Median']:.4f}" if row['Variable'] == 'dlev' else f"{row['Median']:.2f}", axis=1)
    summary_stats['Standard Deviation'] = summary_stats.apply(lambda row: f"{row['Standard Deviation']:.4f}" if row['Variable'] == 'dlev' else f"{row['Standard Deviation']:.2f}", axis=1)
    summary_stats['Variable'] = summary_stats['Variable'].map(variable_labels)
    summary_stats.set_index('Variable', inplace=True)
    print(summary_stats)

    latex_table = tabulate(summary_stats, headers='keys', tablefmt='latex', floatfmt=".2f")
    
    print(latex_table)

def add_stars(tstat):
    if abs(tstat) >= 2.58:
        return '***'
    elif abs(tstat) >= 1.96:
        return '**'
    elif abs(tstat) >= 1.645:
        return '*'
    else:
        return ''

def fm_ret_lev(df, quant_intan, quant_lev, subsample, filename):
    df_copy = df.copy()
    df_copy.reset_index(inplace=True)
    df_copy['year_month'] = df_copy['year_month'].astype(str)
    df_copy['year_month'] = pd.to_datetime(df_copy['year_month'], format='%Y-%m')
    quant_intan_col = f'intan_at_{quant_intan}'
    quant_lev_col = f'lev_{quant_lev}'
    
    if subsample == 'hint':
        # df_subsample = df_copy[df_copy[quant_intan_col] == quant_intan]
        df_subsample = df_copy[(df_copy[quant_intan_col] == quant_intan) & (df_copy[quant_lev_col] == quant_lev)]
        # df_subsample = df_copy[(df_copy[quant_lev_col] == quant_lev)]

    elif subsample == 'lint':
        # df_subsample = df_copy[df_copy[quant_intan_col] == 1]        
        df_subsample = df_copy[(df_copy[quant_intan_col] == 1) & (df_copy[quant_lev_col] == quant_lev)]
        # df_subsample = df_copy[(df_copy[quant_lev_col] == 1)]

    else:
        df_subsample = df_copy
        
    # df_copy['year_month'] = df_copy['year_month'].to_timestamp()
    # df_copy['year_month'] = pd.to_datetime(df_copy['year_month'], format='%Y%m').dt.to_period('M').to_timestamp()
    # df_copy['year_month'] = pd.to_datetime(df_copy['year_month'], format='%Y%m')

    df_subsample.set_index(['GVKEY', 'year_month'], inplace=True)
    
    ###################################
    # Running Fama MacBeth regressions
    ###################################
    # df_copy['quant_intan_dummy'] = df_copy['intan_at_5'] == 5
    # df_copy['dlevXintan_quant'] = df_copy['dlev'] * df_copy['quant_intan_dummy']
    # df_copy['indust_dummy'] = df_copy['ff_indust'] == 10
    # df_copy['dlevXindust_dummy'] = df_copy['dlev'] * df_copy['indust_dummy']
    df_subsample['dummy_intan'] = df_subsample[quant_intan_col] == quant_intan
    df_subsample['dlevXdummy_intan'] = df_subsample['dlev'] * df_subsample['dummy_intan']
    
    # dep_vars = ['RET_lead1', 'ret_2mo_lead1', 'ret_3mo_lead1']
    dep_vars = ['ret_lead1']
    # dep = df_clean_no_na['ret_2mo_lead1']*100
    # indep_vars = ['dlev', 'dlevXindust_dummy', 'indust_dummy', 'lev', 'beta', 'ln_me', 'bm', 'roe', 'one_year_cum_return', 'RET']
    
    indep_vars = ['dlev', 'lev', 'beta', 'ln_me', 'bm', 'roe', 'one_year_cum_return', 'RET']
    # indep_vars = ['dlev', 'dlevXdummy_intan', 'dummy_intan', 'lev', 'beta', 'ln_me', 'bm', 'roe', 'one_year_cum_return', 'RET']

    # indep_vars = ['d_debt_at', 'debt_at', 'beta', 'ln_me', 'bm', 'roe', 'one_year_cum_return', 'RET']
    # indep_vars = ['d_debt_at', 'dummyXd_debt_at', 'lint', 'beta', 'ln_me', 'bm', 'roe', 'one_year_cum_return', 'RET']
    #indep_vars = ['d_ln_debt_at', 'ln_ceqq', 'roa', 'RET', 'beta']
    # indep_vars = ['d_debt_at', 'dummyXd_debt_at', 'lint', 'ln_ceqq', 'roa', 'RET', 'beta']
    # indep_vars = ['d_debt_at', 'dummyXd_debt_at', 'hint', 'ln_ceqq', 'roa', 'RET', 'beta']
    # indep_vars = ['d_debt_at', 'dummyXd_debt_at', 'hint']
    # indep_vars = ['d_debt_at', 'dummyXd_debt_at', 'hint', 'bm']
    # indep_vars = ['d_debt_at', 'ln_ceqq', 'roe', 'RET', 'beta']
    # indep_vars = ['dlev', 'beta', 'ln_me', 'bm', 'roe', 'RET']
    # indep_vars = ['d_ln_debt_at', 'ln_ceqq', 'roa', 'RET', 'bm']

    # Create a dictionary for variable labels
    variable_labels = {
        'd_debt_at': 'Debt/assets Change',
        'd_ln_debt_at': 'Debt/assets log change',
        'debt_at': 'Debt/assets',
        'd_ln_lev': 'Leverage log change',
        'dlev': 'Leverage Change',
        'lev': 'Leverage',
        'mkt_rf': 'Market Risk Premium',
        'smb': 'SMB',
        'hml': 'HML',
        'dummyXd_debt_at': 'High intan/at X Leverage Change',
        'dummyXdebt_at': 'High intan/at X Leverage',    
        'hlev': 'High Leverage dummy',    
        'llev': 'Low Leverage dummy',
        'hint': 'High intangible/assets dummy',
        'lint': 'Low intangible/assets dummy',
        'ln_ceqq': 'Log Equity',
        'ln_me': 'Log Market Value of Equity',
        'ln_at': 'Log Assets',
        'd_roe': 'Return on Equity Change',
        'roe': 'Return on Equity',    
        'roa': 'Return on Assets',
        'one_year_cum_return': 'Previous one-year return',
        'RET': 'Previous month return',
        'beta': 'Beta',
        'bm': 'Book-to-Market Ratio'
    }


    results = {}
    obs_counts = {}
    firm_counts = {}
    r_squared = {}

    for dep_var in dep_vars:
        df_clean_no_na = df_subsample.dropna(subset=[dep_var] + indep_vars)
        dep = df_clean_no_na[dep_var] * 100
        indep = df_clean_no_na[indep_vars]
        mod = FamaMacBeth(dep, indep)
        res = mod.fit()
        results[dep_var] = res
        obs_counts[dep_var] = res.nobs
        df_reset = df_clean_no_na.reset_index()
        firm_counts[dep_var] = df_reset['GVKEY'].nunique()
        r_squared[dep_var] = res.rsquared

    # Extract coefficients, standard errors, and confidence intervals into a DataFrame
    data = {'Variable': indep_vars}
    for dep_var, res in results.items():
        coeffs = res.params.round(2)
        t_stats = res.tstats.round(2)
        std_errors = res.std_errors.round(2)
        conf_int = res.conf_int()
        lower_ci = conf_int.iloc[:, 0].round(2)
        upper_ci = conf_int.iloc[:, 1].round(2)
        
        data[f'{dep_var}_Coeff'] = coeffs
        data[f'{dep_var}_tstat'] = t_stats
        data[f'{dep_var}_SE'] = std_errors
        data[f'{dep_var}_CI_Lower'] = lower_ci
        data[f'{dep_var}_CI_Upper'] = upper_ci

    dataframe = pd.DataFrame(data)
    dataframe['Variable'] = dataframe['Variable'].map(variable_labels)

    # Generate rows for coefficients, standard errors, and confidence intervals
    rows = []
    # for _, row in dataframe.iterrows():
    #     rows.append([row['Variable']] + [f"{row[f'{dep_var}_Coeff']:.4f}" for dep_var in dep_vars])
    #     rows.append([''] + [f"({row[f'{dep_var}_tstat']:.4f})" for dep_var in dep_vars])
    #     rows.append([''] + [f"({row[f'{dep_var}_SE']:.4f})" for dep_var in dep_vars])
    #     rows.append([''] + [f"[{row[f'{dep_var}_CI_Lower']:.4f}, {row[f'{dep_var}_CI_Upper']:.4f}]" for dep_var in dep_vars])
    
    for _, row in dataframe.iterrows():
        coeff_rows = []
        tstat_rows = []
        se_rows = []
        ci_rows = []
        for dep_var in dep_vars:
            coeff = f"{row[f'{dep_var}_Coeff']:.2f}"
            tstat = row[f'{dep_var}_tstat']
            stars = add_stars(tstat)  # Using the existing add_stars function
            coeff_rows.append(f"{coeff}{stars}")
            # tstat_rows.append(f"({tstat:.2f})")
            se_rows.append(f"({row[f'{dep_var}_SE']:.2f})")
            ci_rows.append(f"[{row[f'{dep_var}_CI_Lower']:.2f}, {row[f'{dep_var}_CI_Upper']:.2f}]")



        
        rows.append([row['Variable']] + coeff_rows)
        # rows.append([''] + tstat_rows)
        rows.append([''] + se_rows)
        rows.append([''] + ci_rows)

        # Add rows for number of observations, number of unique firms, and R-squared
        obs_firms_data = [
            ['Observations'] + [obs_counts[dep_var] for dep_var in dep_vars],
            ['Number of Firms'] + [firm_counts[dep_var] for dep_var in dep_vars],
            ['R^{2}'] + [f"{r_squared[dep_var]:.5f}" for dep_var in dep_vars]
        ]

    # Generate LaTeX table with custom header
    custom_header = r"""
\begin{table}[h!]
\centering
\renewcommand{\arraystretch}{1.3} % Increase row height
\begin{tabular}{llll}
\hline
% Independent variables 
& \multicolumn{3}{c}{Stock return (\%)} \\
\cline{2-4}
             & All stocks   & High leverage & High leverage   \\ & & High IK/A & Low IK/A \\
\hline
"""

    # Combine all rows
    all_rows = rows + obs_firms_data

    # Generate the main content of the table
    table_content = tabulate(all_rows, tablefmt='latex_raw', showindex=False)
    
    # Remove the \begin{tabular} and \hline from the generated content
    table_content = table_content.replace(r'\begin{tabular}{l' + 'l'*len(dep_vars) + '}', '')
    table_content = table_content.replace(r'\hline', '', 1)
    
    # Combine custom header with table content
    full_latex_table = custom_header + table_content + "\n" + r"\end{table}"

    # Print LaTeX table
    print(full_latex_table)
    create_latex_table_file(full_latex_table, filename)
    
# def fm_ret_lev(df, quant_intan, subsample, filename):
#     df_copy = df.copy()
#     df_copy.reset_index(inplace=True)
#     df_copy['year_month'] = df_copy['year_month'].astype(str)
#     df_copy['year_month'] = pd.to_datetime(df_copy['year_month'], format='%Y-%m')
#     quant_intan_col = f'intan_at_{quant_intan}'
    
#     if subsample == 'hint':
#         df_subsample = df_copy[df_copy[quant_intan_col] == quant_intan]
#     elif subsample == 'lint':
#         df_subsample = df_copy[df_copy[quant_intan_col] == 1]
#     else:
#         df_subsample = df_copy
        
#     # df_copy['year_month'] = df_copy['year_month'].to_timestamp()
#     # df_copy['year_month'] = pd.to_datetime(df_copy['year_month'], format='%Y%m').dt.to_period('M').to_timestamp()
#     # df_copy['year_month'] = pd.to_datetime(df_copy['year_month'], format='%Y%m')

#     df_subsample.set_index(['GVKEY', 'year_month'], inplace=True)
    
#     ###################################
#     # Running Fama MacBeth regressions
#     ###################################
#     # df_copy['quant_intan_dummy'] = df_copy['intan_at_5'] == 5
#     # df_copy['dlevXintan_quant'] = df_copy['dlev'] * df_copy['quant_intan_dummy']
#     # df_copy['indust_dummy'] = df_copy['ff_indust'] == 10
#     # df_copy['dlevXindust_dummy'] = df_copy['dlev'] * df_copy['indust_dummy']
#     df_subsample['dummy_intan'] = df_subsample[quant_intan_col] == quant_intan
#     df_subsample['dlevXdummy_intan'] = df_subsample['dlev'] * df_subsample['dummy_intan']
    
#     # dep_vars = ['RET_lead1', 'ret_2mo_lead1', 'ret_3mo_lead1']
#     dep_vars = ['ret_lead1']
#     # dep = df_clean_no_na['ret_2mo_lead1']*100
#     # indep_vars = ['dlev', 'dlevXindust_dummy', 'indust_dummy', 'lev', 'beta', 'ln_me', 'bm', 'roe', 'one_year_cum_return', 'RET']
    
#     indep_vars = ['dlev', 'lev', 'beta', 'ln_me', 'bm', 'roe', 'one_year_cum_return', 'RET']
#     # indep_vars = ['dlev', 'dlevXdummy_intan', 'dummy_intan', 'lev', 'beta', 'ln_me', 'bm', 'roe', 'one_year_cum_return', 'RET']

#     # indep_vars = ['d_debt_at', 'debt_at', 'beta', 'ln_me', 'bm', 'roe', 'one_year_cum_return', 'RET']
#     # indep_vars = ['d_debt_at', 'dummyXd_debt_at', 'lint', 'beta', 'ln_me', 'bm', 'roe', 'one_year_cum_return', 'RET']
#     #indep_vars = ['d_ln_debt_at', 'ln_ceqq', 'roa', 'RET', 'beta']
#     # indep_vars = ['d_debt_at', 'dummyXd_debt_at', 'lint', 'ln_ceqq', 'roa', 'RET', 'beta']
#     # indep_vars = ['d_debt_at', 'dummyXd_debt_at', 'hint', 'ln_ceqq', 'roa', 'RET', 'beta']
#     # indep_vars = ['d_debt_at', 'dummyXd_debt_at', 'hint']
#     # indep_vars = ['d_debt_at', 'dummyXd_debt_at', 'hint', 'bm']
#     # indep_vars = ['d_debt_at', 'ln_ceqq', 'roe', 'RET', 'beta']
#     # indep_vars = ['dlev', 'beta', 'ln_me', 'bm', 'roe', 'RET']
#     # indep_vars = ['d_ln_debt_at', 'ln_ceqq', 'roa', 'RET', 'bm']

#     # Create a dictionary for variable labels
#     variable_labels = {
#         'd_debt_at': 'Debt/assets Change',
#         'd_ln_debt_at': 'Debt/assets log change',
#         'debt_at': 'Debt/assets',
#         'd_ln_lev': 'Leverage log change',
#         'dlev': 'Leverage Change',
#         'lev': 'Leverage',
#         'mkt_rf': 'Market Risk Premium',
#         'smb': 'SMB',
#         'hml': 'HML',
#         'dummyXd_debt_at': 'High intan/at X Leverage Change',
#         'dummyXdebt_at': 'High intan/at X Leverage',    
#         'hlev': 'High Leverage dummy',    
#         'llev': 'Low Leverage dummy',
#         'hint': 'High intangible/assets dummy',
#         'lint': 'Low intangible/assets dummy',
#         'ln_ceqq': 'Log Equity',
#         'ln_me': 'Log Market Value of Equity',
#         'ln_at': 'Log Assets',
#         'd_roe': 'Return on Equity Change',
#         'roe': 'Return on Equity',    
#         'roa': 'Return on Assets',
#         'one_year_cum_return': 'Previous one-year return',
#         'RET': 'Previous month return',
#         'beta': 'Beta',
#         'bm': 'Book-to-Market Ratio'
#     }


#     results = {}
#     obs_counts = {}
#     firm_counts = {}
#     r_squared = {}

#     for dep_var in dep_vars:
#         df_clean_no_na = df_subsample.dropna(subset=[dep_var] + indep_vars)
#         dep = df_clean_no_na[dep_var] * 100
#         indep = df_clean_no_na[indep_vars]
#         mod = FamaMacBeth(dep, indep)
#         res = mod.fit()
#         results[dep_var] = res
#         obs_counts[dep_var] = res.nobs
#         df_reset = df_clean_no_na.reset_index()
#         firm_counts[dep_var] = df_reset['GVKEY'].nunique()
#         r_squared[dep_var] = res.rsquared

#     # Extract coefficients and t-stats into a DataFrame
#     data = {'Variable': indep_vars}
#     for dep_var, res in results.items():
#         coeffs = res.params.round(2).astype(str) + res.tstats.apply(add_stars)
#         tstats = res.tstats.round(2).astype(str)
#         data[f'{dep_var}_Coeff'] = coeffs
#         data[f'{dep_var}_t'] = tstats

#     dataframe = pd.DataFrame(data)
#     dataframe['Variable'] = dataframe['Variable'].map(variable_labels)

#     # Generate rows for coefficients and t-stats
#     rows = []
#     for _, row in dataframe.iterrows():
#         rows.append([row['Variable']] + [row[f'{dep_var}_Coeff'] for dep_var in dep_vars])
#         rows.append([''] + [f"({row[f'{dep_var}_t']})" for dep_var in dep_vars])

#     # Add rows for number of observations, number of unique firms, and R-squared
#     obs_firms_data = [
#         ['Observations'] + [obs_counts[dep_var] for dep_var in dep_vars],
#         ['Number of Firms'] + [firm_counts[dep_var] for dep_var in dep_vars],
#         ['R^{2}'] + [round(r_squared[dep_var], 5) for dep_var in dep_vars]
#     ]

#     # Generate LaTeX table with custom header
#     custom_header = r"""
# \begin{table}[h!]
# \centering
# \renewcommand{\arraystretch}{1.5} % Increase row height
# \begin{tabular}{llll}
# \hline
# % Independent variables 
# & \multicolumn{3}{c}{Stock return (\%)} \\
# \cline{2-4}
#              & All stocks   & High IK/A  & Low IK/A      \\
# \hline
# """

#     # # Combine all rows
#     all_rows = rows + obs_firms_data

#     # # Generate LaTeX table
#     # latex_table = tabulate(all_rows, headers=['Variable'] + dep_vars, tablefmt='latex_raw', showindex=False)


#     # Generate the main content of the table
#     table_content = tabulate(all_rows, tablefmt='latex_raw', showindex=False)
    
#     # Remove the \begin{tabular} and \hline from the generated content
#     table_content = table_content.replace(r'\begin{tabular}{l' + 'l'*len(dep_vars) + '}', '')
#     table_content = table_content.replace(r'\hline', '', 1)
    
#     # Combine custom header with table content
#     full_latex_table = custom_header + table_content + "\n" + r"\end{table}"

#     # Print LaTeX table
#     print(full_latex_table)
#     create_latex_table_file(full_latex_table, filename)
    

    #######################################################
    # To remove R2, replace the above with the following:
    #######################################################

    # results = {}
    # obs_counts = {}
    # firm_counts = {}

    # for dep_var in dep_vars:
    #     df_clean_no_na = df.dropna(subset=[dep_var] + indep_vars)
    #     dep = df_clean_no_na[dep_var] * 100
    #     indep = df_clean_no_na[indep_vars]
    #     mod = FamaMacBeth(dep, indep)
    #     res = mod.fit()
    #     results[dep_var] = res
    #     obs_counts[dep_var] = res.nobs
    #     df_reset = df_clean_no_na.reset_index()
    #     firm_counts[dep_var] = df_reset['GVKEY'].nunique()

    # # Extract coefficients and t-stats into a DataFrame
    # data = {'Variable': indep_vars}
    # for dep_var, res in results.items():
    #     coeffs = res.params.round(2).astype(str) + res.tstats.apply(add_stars)
    #     tstats = res.tstats.round(2).astype(str)
    #     data[f'{dep_var}_Coeff'] = coeffs
    #     data[f'{dep_var}_t'] = tstats

    # dataframe = pd.DataFrame(data)
    # dataframe['Variable'] = dataframe['Variable'].map(variable_labels)

    # # Generate rows for coefficients and t-stats
    # rows = []
    # for _, row in dataframe.iterrows():
    #     rows.append([row['Variable']] + [row[f'{dep_var}_Coeff'] for dep_var in dep_vars])
    #     rows.append([''] + [f"({row[f'{dep_var}_t']})" for dep_var in dep_vars])

    # # Add rows for number of observations and number of unique firms
    # obs_firms_data = [
    #     ['Observations'] + [obs_counts[dep_var] for dep_var in dep_vars],
    #     ['Number of Firms'] + [firm_counts[dep_var] for dep_var in dep_vars]
    # ]

    # # Combine all rows
    # all_rows = rows + obs_firms_data

    # # Generate LaTeX table
    # latex_table = tabulate(all_rows, headers=['Variable'] + dep_vars, tablefmt='latex_raw', showindex=False)

    # # Print LaTeX table
    # print(latex_table)

    # Function to perform regression on a constant
def regress_on_constant(df):
    df = df.dropna(subset=['ret_diff'])  # Remove rows with NaN values in 'RET_diff'
    X = sm.add_constant(pd.Series([1]*len(df), index=df.index))  # Add a constant term (array of ones) with the same index as df
    model = sm.OLS(df['ret_diff'], X).fit()
    return model

def create_latex_table_file(latex_table, filename):
    """
    Writes the LaTeX table to a .tex file.

    Parameters:
    latex_table (str): The LaTeX table string.
    filename (str): The name of the .tex file to save the table.
    """
    # Get the folder path where the file should be saved
    texfolder = ut.to_output('tex/')
    
    # Combine the folder path with the filename
    filepath = texfolder + filename
    
    # Write the LaTeX table to the specified file
    with open(filepath, "w") as tex_file:
        tex_file.write(latex_table)

def idl_factor(df, quant_intan, quant_dlev, quant_size, quant_bm):
    df_copy = df.copy()
    quant_intan_col = f'intan_at_{quant_intan}'
    quant_dlev_col = f'dlev_{quant_dlev}'
    quant_size_col = f'size_{quant_size}'
    quant_bm_col = f'bm_{quant_bm}'
    
    ##########################################################################
    # Create a new asset pricing factor (based on intangible/assets and dlev)
    ##########################################################################
    grouped = df_copy.groupby(['year_month', quant_intan_col, quant_dlev_col, quant_size_col])
    # grouped = df_copy.groupby(['year_month', quant_size_col])
    avg_returns = grouped['ret_lead1'].mean().reset_index()
    
    # Group by year_month and calculate the averages for each condition    
    monthly_avg_returns = avg_returns.groupby('year_month').apply(lambda x: pd.Series({
        'avg_hint_ldl_s': x[(x[quant_intan_col] == quant_intan) & (x[quant_dlev_col] == 1) & (x[quant_size_col] == 1)]['ret_lead1'].mean(),
        'avg_hint_ldl_b': x[(x[quant_intan_col] == quant_intan) & (x[quant_dlev_col] == 1) & (x[quant_size_col] == quant_size)]['ret_lead1'].mean(),
        
        'avg_lint_hdl_s': x[(x[quant_intan_col] == 1) & (x[quant_dlev_col] == quant_dlev) & (x[quant_size_col] == 1)]['ret_lead1'].mean(),
        'avg_lint_hdl_b': x[(x[quant_intan_col] == 1) & (x[quant_dlev_col] == quant_dlev) & (x[quant_size_col] == quant_size)]['ret_lead1'].mean()
    }))
    
    # monthly_avg_returns = avg_returns.groupby('year_month').apply(lambda x: pd.Series({
    #     'avg_hint_ldl_h': x[(x[quant_intan_col] == quant_intan) & (x[quant_dlev_col] == 1) & (x[quant_bm_col] == quant_bm)]['ret_lead1'].mean(),
    #     'avg_hint_ldl_m': x[(x[quant_intan_col] == quant_intan) & (x[quant_dlev_col] == 1) & (x[quant_bm_col] == 2)]['ret_lead1'].mean(),
    #     'avg_hint_ldl_l': x[(x[quant_intan_col] == quant_intan) & (x[quant_dlev_col] == 1) & (x[quant_bm_col] == 1)]['ret_lead1'].mean(),
                
    #     'avg_lint_hdl_h': x[(x[quant_intan_col] == 1) & (x[quant_dlev_col] == quant_dlev) & (x[quant_bm_col] == quant_bm)]['ret_lead1'].mean(),
    #     'avg_lint_hdl_m': x[(x[quant_intan_col] == 1) & (x[quant_dlev_col] == quant_dlev) & (x[quant_bm_col] == 2)]['ret_lead1'].mean(),
    #     'avg_lint_hdl_l': x[(x[quant_intan_col] == 1) & (x[quant_dlev_col] == quant_dlev) & (x[quant_bm_col] == 1)]['ret_lead1'].mean()
    # }))
        
    # monthly_avg_returns = avg_returns.groupby('year_month').apply(lambda x: pd.Series({
    #     'small': x[(x[quant_size_col] == 1)]['ret_lead1'].mean(),
    #     'big': x[(x[quant_size_col] == quant_size)]['ret_lead1'].mean()
    # }))
    
    # Calculate the difference
    # monthly_avg_returns['difference'] = monthly_avg_returns['small'] - monthly_avg_returns['big']
    monthly_avg_returns['difference'] = ((monthly_avg_returns['avg_hint_ldl_s'] + monthly_avg_returns['avg_hint_ldl_b'])/2) - ((monthly_avg_returns['avg_lint_hdl_s'] + monthly_avg_returns['avg_lint_hdl_b'])/2)
    # monthly_avg_returns['difference'] = ((monthly_avg_returns['avg_lint_hdl_h'] + monthly_avg_returns['avg_lint_hdl_m'] + monthly_avg_returns['avg_hint_ldl_l'])/3) - ((monthly_avg_returns['avg_lint_hdl_h'] + monthly_avg_returns['avg_lint_hdl_m'] + monthly_avg_returns['avg_lint_hdl_l'])/3)


    # Create the final time series
    time_series = monthly_avg_returns['difference'].reset_index()
    time_series.columns = ['year_month', 'idl']
    
    df_factor = pd.merge(df_copy, time_series, on='year_month', how='left')
    df_factor['idl'] = df_factor['idl']*100 # Convert to percentage (and make it consistent with the other factors)    
    
    return df_factor


# def time_series_regression(df, firm_id):
#     # Filter data for a single firm
#     firm_data = df[df['GVKEY'] == firm_id]
    
#     # Prepare X and y
#     X = sm.add_constant(firm_data[['mkt_rf', 'hml', 'smb']])
#     y = firm_data['ret_lead1']*100  # Convert to percentage
    
#     # Run time-series regression
#     model = sm.OLS(y, X).fit()
    
#     return pd.Series({'firm_id': firm_id, 'const': model.params['const'], 
#                       'mkt_rf_beta': model.params['mkt_rf'], 
#                       'hml_beta': model.params['hml'], 
#                       'smb_beta': model.params['smb']})


# def fama_macbeth_regression(df):
#     df = df.reset_index()
#     # Get unique firm IDs
#     firms = df['GVKEY'].unique()
    
#     # Run time-series regressions for each firm
#     firm_betas = []
#     for firm in firms:
#         try:
#             betas = time_series_regression(df, firm)
#             firm_betas.append(betas)
#         except:
#             print(f"Regression failed for firm {firm}")
    
#     # Convert to DataFrame
#     betas_df = pd.DataFrame(firm_betas)
#     betas_df = betas_df.rename(columns={'const': 'alpha', 'mkt_rf': 'mkt_rf_beta', 'hml': 'hml_beta', 'smb': 'smb_beta'})
    
    
#     # Second stage: cross-sectional regression
#     # Here we would regress returns on the estimated betas
#     # This part depends on how you want to structure your second stage
    
#     return betas_df


# def second_stage_regression(df, betas):
#     """
#     Perform the second stage regression for a single time period.
#     """
#     returns = df['ret_lead1']*100  # Convert to percentage
#     X = sm.add_constant(betas)
#     y = returns
#     model = sm.OLS(y, X).fit()
#     return model.params


# def fama_macbeth_second_stage(df, betas_df):
#     """
#     Perform the second stage of the Fama-MacBeth procedure.
#     """
#     # Merge returns with betas
#     df = df.reset_index()
#     returns_df = df[['GVKEY', 'ret_lead1']]*100  # Convert to percentage
#     merged_df = returns_df.merge(betas_df, on='GVKEY', how='inner')
    
#     # Group by date
#     grouped = merged_df.groupby('year_month')
    
#     # Run cross-sectional regressions for each time period
#     results = []
#     for date, group in grouped:
#         try:
#             params = second_stage_regression(group['ret_lead1'], 
#                                              group[['mkt_rf_beta', 'hml_beta', 'smb_beta']], 
#                                              date)
#             results.append(params)
#         except:
#             print(f"Regression failed for date {date}")
    
#     # Convert results to DataFrame
#     results_df = pd.DataFrame(results)
    
#     # Calculate means and standard errors
#     means = results_df.mean()
#     std_errors = results_df.std() / np.sqrt(len(results_df))
#     t_stats = means / std_errors
#     p_values = 2 * (1 - stats.t.cdf(abs(t_stats), df=len(results_df)-1))
    
#     # Combine results
#     summary = pd.DataFrame({
#         'Coefficient': means,
#         'Std Error': std_errors,
#         't-statistic': t_stats,
#         'p-value': p_values
#     })
    
#     return summary

def reg_factor(df, quant_intan, quant_dlev, quant_lev, quant_size, quant_bm, window, model_factor, filename):
    df_copy = df.copy()
    df_copy.reset_index(inplace=True)
    df_copy['year_month'] = df_copy['year_month'].astype(str)
    df_copy['year_month'] = pd.to_datetime(df_copy['year_month'], format='%Y-%m')
    # df_copy.set_index(['GVKEY', 'year_month'], inplace=True)
    # df_copy = df_copy.set_index(['GVKEY', 'year_month'])
    df_copy['const'] = 1 # Add a constant term    
    quant_intan_col = f'intan_at_{quant_intan}'
    quant_dlev_col = f'dlev_{quant_dlev}'
    quant_lev_col = f'lev_{quant_lev}'
    quant_size_col = f'size_{quant_size}'
    quant_bm_col = f'bm_{quant_bm}'
    
    ##########################################
    # Change here to use different portfolios and factors
    ##########################################
    
    portfolio_cols = ['year_month', quant_size_col, quant_intan_col, quant_dlev_col]
    factors_list = ['idl', 'mkt_rf', 'hml', 'smb', 'mom', 'cma', 'rmw']

    # factors_list = ['mkt_rf', 'hml', 'smb', 'mom']
    
    # portfolio_cols = ['year_month', quant_size_col, quant_bm_col]
    # factors_list = ['mkt_rf', 'hml', 'smb']
    
    
    ##########################################
    
    df_copy = (df_copy
               .assign(ret_lead1_rf = df_copy['ret_lead1'] - (df_copy['rf']/100),
                       ret_lead2_rf = df_copy['ret_lead2'] - (df_copy['rf']/100))
    )
    
    factors = df_copy[['year_month', 'const'] + factors_list].drop_duplicates()
    
    ######################################################################################
    # Creating the portfolio returns (3 intan x 5 dlev x 4 lev = 60 portfolios per period)
    ######################################################################################
    port_returns = df_copy.groupby(portfolio_cols)[['ret_lead1_rf', 'ret_lead2_rf']].mean().reset_index()
    # port_returns = df_copy.groupby(['year_month', quant_size_col])[['ret_lead1_rf', 'ret_lead2_rf']].mean().reset_index()

    #####################################
    # UNCOMMENT HERE TO USE FF PORTFOLIOS
    #####################################
    
    # port_ff = (port_ff
    #            .assign(ret_lead1_rf = port_ff.groupby('port_id')['ret'].shift(-1))
    #            )
    
    # df_port = port_ff.merge(factors, on=['year_month'], how='left')
    # # df_port = (df_port
    # #     .assign(const = lambda x: pd.to_numeric(x['const'], errors='coerce').astype('Int64'))
    # # )
    
    # df_port = df_port.dropna(subset=factors_list)

    #####################################
    #####################################
    
    #####################################
    # UNCOMMENT TO CREATE OWN PORTFOLIOS
    #####################################
    df_port = port_returns.merge(factors, on=['year_month'], how='inner')
    # df_port['port_id'] = df_port[quant_dlev_col].astype(str)
    # df_port['port_id'] = df_port[quant_intan_col].astype(str) + df_port[quant_dlev_col].astype(str) + df_port[quant_size_col].astype(str)
    id_cols = [col for col in portfolio_cols if col != 'year_month']
    df_port['port_id'] = df_port[id_cols].astype(str).agg(''.join, axis=1)
    # df_port['port_id'] = df_port[quant_size_col].astype(str)
    df_port['port_id'] = df_port['port_id'].astype('Int64')
    
    #####################################
    #####################################
    
    df_betas = create_factor_betas(df_port, factors_list, window) # this creates the factor betas for each portfolio
    df_returns = df_port[['port_id', 'year_month', 'ret_lead1_rf']]

    # df_returns = df_port[['port_id', 'year_month', 'ret_lead1', 'ret_lead2']]
    df_fm = df_returns.merge(df_betas, on=['port_id', 'year_month'], how='inner')
    df_fm['const'] = 1
    # df_fm = (df_fm
    #     .assign(const = lambda x: pd.to_numeric(x['const'], errors='coerce').astype('Int64'))
    # )
    
    df_fm.set_index(['port_id', 'year_month'], inplace=True)

    dep_vars = ['ret_lead1_rf']

    if model_factor == 'capm':
        indep_vars = ['beta_idl', 'beta_mkt', 'const'] # CAPM
        # indep_vars = ['beta_mkt', 'const'] # CAPM
    elif model_factor == 'ff3':
        indep_vars = ['beta_idl', 'beta_mkt', 'beta_hml', 'beta_smb', 'const'] # Fama-French 3-factor
        # indep_vars = ['beta_mkt', 'beta_hml', 'beta_smb', 'const'] # Fama-French 3-factor
    elif model_factor == 'carhart4':
        indep_vars = ['beta_idl', 'beta_mkt', 'beta_hml', 'beta_smb', 'beta_mom', 'const'] # Cahart 4-factor
        # indep_vars = ['beta_mkt', 'beta_hml', 'beta_smb', 'beta_mom', 'const'] # Cahart 4-factor
    elif model_factor == 'ff5':
        indep_vars = ['beta_idl', 'beta_mkt', 'beta_hml', 'beta_smb', 'beta_cma', 'beta_rmw', 'const'] # Fama-French 5-factor
        # indep_vars = ['beta_mkt', 'beta_hml', 'beta_smb', 'beta_cma', 'beta_rmw', 'const'] # Fama-French 5-factor
    else:
        raise ValueError(f"Invalid model_factor: {model_factor}. Please choose from 'capm', 'ff3', 'carhart4', or 'ff5'.")

    # indep_vars = ['beta_mkt', 'beta_hml', 'beta_smb', 'const'] # Fama-French 5-factor


    # indep_vars = ['beta_mkt', 'beta_hml', 'beta_smb', 'const']
    # indep_vars = ['beta_smb', 'const']


    # indep_vars = ['beta_mkt', 'const']

    # indep_vars = ['dlev', 'hml', 'smb', 'bm', 'roe', 'one_year_cum_return', 'RET']

    # Create a dictionary for variable labels
    variable_labels = {
        'beta_idl': 'IDL',
        'beta_mkt': 'Mkt-RF',
        'beta_hml': 'HML',
        'beta_smb': 'SMB',
        'beta_cma': 'CMA',
        'beta_rmw': 'RMW',
        'beta_mom': 'MOM',
        'const': 'Intercept'
    }

    results = {}
    obs_counts = {}
    port_counts = {}
    r_squared = {}
    
    for dep_var in dep_vars:
        df_clean_no_na = df_fm.dropna(subset=[dep_var] + indep_vars)
        dep = df_clean_no_na[dep_var] * 100
        indep = df_clean_no_na[indep_vars]
        indep = indep.astype(float)
        mod = FamaMacBeth(dep, indep)
        res = mod.fit(cov_type='robust')
        results[dep_var] = res
        obs_counts[dep_var] = res.nobs
        df_reset = df_clean_no_na.reset_index()
        port_counts[dep_var] = df_reset['port_id'].nunique()
        r_squared[dep_var] = res.rsquared_between

    # Extract coefficients and t-stats into a DataFrame
    data = {'Variable': indep_vars}
    for dep_var, res in results.items():
        coeffs = res.params.round(2).astype(str) + res.tstats.apply(add_stars)
        tstats = res.tstats.round(2).astype(str)
        data[f'{dep_var}_Coeff'] = coeffs
        data[f'{dep_var}_t'] = tstats

    dataframe = pd.DataFrame(data)
    dataframe['Variable'] = dataframe['Variable'].map(variable_labels)

    # Generate rows for coefficients and t-stats
    rows = []
    for _, row in dataframe.iterrows():
        rows.append([row['Variable']] + [row[f'{dep_var}_Coeff'] for dep_var in dep_vars])
        rows.append([''] + [f"({row[f'{dep_var}_t']})" for dep_var in dep_vars])

    # Add a horizontal line
    # rows.append([r'\hline'] * (len(dep_vars) + 1))

    # Add rows for number of observations, number of unique firms, and R-squared
    obs_firms_data = [
        ['Observations'] + [obs_counts[dep_var] for dep_var in dep_vars],
        ['Number of portfolios'] + [port_counts[dep_var] for dep_var in dep_vars],
        ['R^{2}'] + [round(r_squared[dep_var], 5) for dep_var in dep_vars]
    ]

    # Generate LaTeX table with custom header
    custom_header = r"""
\begin{table}[h!]
\centering
\renewcommand{\arraystretch}{1.5} % Increase row height
\begin{tabular}{lllll}
\hline
% Independent variables 
% & \multicolumn{2}{c}{Stock return (\%)} \\
% \cline{2-3}
            & CAPM+
             & FF3+           & Carhart4+    
             & FF5+ \\
\hline
"""

    # # Combine all rows
    all_rows = rows + obs_firms_data

    # # Generate LaTeX table
    # latex_table = tabulate(all_rows, headers=['Variable'] + dep_vars, tablefmt='latex_raw', showindex=False)


    # Generate the main content of the table
    table_content = tabulate(all_rows, tablefmt='latex_raw', showindex=False)
    
    # Remove the \begin{tabular} and \hline from the generated content
    table_content = table_content.replace(r'\begin{tabular}{l' + 'l'*len(dep_vars) + '}', '')
    table_content = table_content.replace(r'\hline', '', 1)
    
    # Combine custom header with table content
    full_latex_table = custom_header + table_content + "\n" + r"\end{table}"

    # Print LaTeX table
    print(full_latex_table)
    create_latex_table_file(full_latex_table, filename)
    

def create_factor_betas(df, factors_list, window):
    """
    Create factor betas for each portfolio.
    
    Parameters:
    df (DataFrame): The DataFrame containing the portfolios and factor returns.
    window (int): The number of months to use in the rolling window.
    
    Returns:
    DataFrame: A DataFrame containing the factor betas for each portfolio and time period.
    """
    # Ensure the DataFrame is sorted by port_id and year_month
    df = df.sort_values(['port_id', 'year_month'])

    # Create a list to store the factor betas
    factor_betas = []

    # Get the unique portfolio IDs
    port_ids = df['port_id'].unique()

    # Loop through each portfolio
    for port_id in port_ids:
        # Filter the DataFrame for the current portfolio
        port_data = df[df['port_id'] == port_id].copy()
        
        # Ensure the data is sorted by date
        port_data = port_data.sort_values('year_month')
        
        # Prepare the data for rolling regression
        X = port_data[factors_list]
        y = port_data['ret_lead1_rf'] * 100  # Convert to percentage
        
        # Perform rolling regression
        model = RollingOLS(y, sm.add_constant(X), window=window)
        results = model.fit()
        
        # Extract the parameters (betas)
        betas = results.params.iloc[window-1:]  # Start from the window'th observation
        
        # Add portfolio ID and reset the index to get the date
        betas['port_id'] = port_id
        betas['year_month'] = port_data['year_month'].iloc[window-1:]
        betas = betas.reset_index()

        # Rename columns for clarity
        betas = (betas
                 .rename(columns={#'const': 'beta_const', 
                                      'idl': 'beta_idl', 
                                      'mkt_rf': 'beta_mkt', 
                                      'hml': 'beta_hml', 
                                      'smb': 'beta_smb',
                                      'cma': 'beta_cma',
                                      'rmw': 'beta_rmw',
                                      'mom': 'beta_mom'
                                      })
                 .drop(columns='const')
                 )
        
        # Append to the list
        factor_betas.append(betas)

    # Concatenate all the DataFrames in the list
    factor_betas_df = pd.concat(factor_betas, ignore_index=True)
    
    # Take year_month from port_data considering the window
    # factor_betas_df['year_month'] = port_data['year_month'].iloc[window-1:]
    
    # Ensure the year_month column is datetime
    # factor_betas_df['year_month'] = pd.to_datetime(factor_betas_df['index'])
    factor_betas_df = factor_betas_df.drop('index', axis=1)
    
    return factor_betas_df

# def create_factor_betas(df, window):
#     """
#     Create factor betas for each portfolio.
    
#     Parameters:
#     df (DataFrame): The DataFrame containing the portfolios and factor returns.
#     window (int): The number of months to use in the rolling window.
    
#     Returns:
#     DataFrame: A DataFrame containing the factor betas for each portfolio.
#     """
#     # Create a list to store the factor betas
#     factor_betas = []
    
#     # Get the unique portfolio IDs
#     port_ids = df['port_id'].unique()
    
#     # Create a list to store the factor betas for the current portfolio
#     port_factor_betas = []
    
#     # Loop through each portfolio
#     for port_id in port_ids:
#         # Filter the DataFrame for the current portfolio
#         port_data = df[df['port_id'] == port_id]
        
#         X = port_data[['const', 'idl', 'mkt_rf', 'hml', 'smb']]
#         y = port_data['ret_lead1']*100  # Convert to percentage
        
#         model = RollingOLS(y, X, window=window)
#         model = model.fit()
#         port_factor_betas = model.params

#         # Create a DataFrame with the factor betas for the current portfolio
#         port_factor_betas_df = pd.DataFrame(port_factor_betas, columns=['idl', 'mkt_rf', 'hml', 'smb'])
        
#         # Add the portfolio ID to the DataFrame
#         port_factor_betas_df['port_id'] = port_id
        
    
#     # Append the DataFrame to the list
#     factor_betas.append(port_factor_betas_df)
    
#     # Concatenate all the DataFrames in the list
#     factor_betas_df = pd.concat(factor_betas)
    
#     return factor_betas_df
               
    
def create_dlev_intan_port(df, quant_dlev, quant_intan, quant_pd, pd_subsample, filename):
    df_copy = df.copy()
    quant_dlev_col = f'dlev_{quant_dlev}'
    quant_intan_col = f'intan_at_{quant_intan}'
    quant_pd_col = f'pd_{quant_pd}'
    
    if pd_subsample == 'all':
        df_subsample = df_copy
    elif pd_subsample == 'hpd':
        df_subsample = df_copy[df_copy[quant_pd_col] == quant_pd]
    elif pd_subsample == 'lpd':
        df_subsample = df_copy[df_copy[quant_pd_col] == 1]
    else:
        raise ValueError(f"Invalid pd_subsample: {pd_subsample}. Please choose from 'all', 'hpd', or 'lpd'.")
    # df_subsample = df_copy

    grouped = df_subsample.groupby(['year_month', quant_intan_col, quant_dlev_col])

    # Count the number of stocks in each group
    group_counts = grouped.size().reset_index(name='counts')

    # Filter out the groups that have fewer than 30 stocks
    filtered_groups = group_counts[group_counts['counts'] >= 30]

    # Merge the filtered groups back to the original DataFrame to keep only the valid groups
    filtered_df = df_subsample.merge(filtered_groups.drop('counts', axis=1), 
                                on=['year_month', quant_intan_col, quant_dlev_col], 
                                how='inner')

    # Re-group if necessary for further processing
    filtered_grouped = filtered_df.groupby(['year_month', quant_intan_col, quant_dlev_col])

    columns_to_process = ['ret_lead1', 'ret_lead2']
    
    latex_output = """
\\begin{table}    
\\centering
\\small
\\setlength{\\tabcolsep}{3pt} % Adjust horizontal spacing
\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l*{2}{c}l*{2}{c}}
\\toprule
& \multicolumn{5}{c}{Intangible/assets portfolio} \\\\
\\cmidrule(r){2-3} \\cmidrule(l){4-6}
Leverage & \\multicolumn{2}{c}{Equal-weighted} & & \\multicolumn{2}{c}{Value-weighted} \\\\
\\cmidrule(r){2-3} \\cmidrule(l){4-6}
change portfolio & 1 (Lowest) & 3 (Highest) & & 1 (Lowest) & 3 (Highest) \\\\
\\midrule
Panel A: 1-month ahead average returns & & & & & \\\\
\\midrule
"""

    for idx, col in enumerate(columns_to_process):
        # Equal-weighted average returns
        monthly_avg_returns = filtered_grouped[col].mean().reset_index()

        # Weighted average returns
        monthly_weighted_avg_returns = filtered_grouped.apply(lambda x: (x[col] * x['me']).sum() / x['me'].sum()).reset_index(name=f'weighted_{col}')
        
        # Compute the Average Across All Months for equal-weighted
        overall_avg_returns = monthly_avg_returns.groupby([quant_intan_col, quant_dlev_col])[col].mean().reset_index()

        # Compute the Average Across All Months for weighted average
        overall_weighted_avg_returns = monthly_weighted_avg_returns.groupby([quant_intan_col, quant_dlev_col])[f'weighted_{col}'].mean().reset_index()

        # Pivot the DataFrame to get the correct format for LaTeX
        pivot_df = overall_avg_returns.pivot(index=quant_dlev_col, columns=quant_intan_col, values=col)
        pivot_weighted_df = overall_weighted_avg_returns.pivot(index=quant_dlev_col, columns=quant_intan_col, values=f'weighted_{col}')

        # Calculate the difference between the first and last rows
        difference_row = pivot_df.iloc[0] - pivot_df.iloc[-1]
        difference_row.name = f'1-{quant_dlev} Difference'

        weighted_difference_row = pivot_weighted_df.iloc[0] - pivot_weighted_df.iloc[-1]
        weighted_difference_row.name = f'1-{quant_dlev} Difference'

        # Append the difference row to the pivoted DataFrame
        pivot_df = pivot_df._append(difference_row)
        pivot_weighted_df = pivot_weighted_df._append(weighted_difference_row)

        ######################
        # Calculating std errors and CIs (Equal-weighted)
        ######################

        pivot_stat = monthly_avg_returns.pivot_table(values=col, index=['year_month', quant_intan_col], columns=quant_dlev_col).reset_index()
        pivot_stat['ret_diff'] = pivot_stat[1] - pivot_stat[quant_dlev]
        time_series_diff = pivot_stat[['year_month', quant_intan_col, 'ret_diff']]

        std_errors = []
        confidence_intervals = []
        p_values = []
        t_stats = []

        # Loop through first and last quant_intan values and perform the regression
        for quant_intan_value in [time_series_diff[quant_intan_col].min(), time_series_diff[quant_intan_col].max()]:
            df_filtered = time_series_diff[time_series_diff[quant_intan_col] == quant_intan_value]
            model = regress_on_constant(df_filtered)
            t_stats.append(model.tvalues[0])
            std_errors.append(model.bse[0]*100)
            ci = model.conf_int().iloc[0]*100
            confidence_intervals.append((ci[0], ci[1]))
            p_values.append(model.pvalues[0])

        # t_stats_row = pd.DataFrame([t_stats], columns=time_series_diff[quant_intan_col].unique())
        t_stats_row = pd.DataFrame([t_stats], columns=[time_series_diff[quant_intan_col].min(), time_series_diff[quant_intan_col].max()])
        std_errors_row = pd.DataFrame([std_errors], columns=[time_series_diff[quant_intan_col].min(), time_series_diff[quant_intan_col].max()])

        pivot_df_percent = pivot_df * 100
        
        pivot_df_with_t_stats = pivot_df_percent._append(t_stats_row, ignore_index=True)

        ######################
        # Calculating std errors and CIs (Weighted)
        ######################

        pivot_weighted_stat = monthly_weighted_avg_returns.pivot_table(values=f'weighted_{col}', index=['year_month', quant_intan_col], columns=quant_dlev_col).reset_index()
        pivot_weighted_stat['ret_diff'] = pivot_weighted_stat[1] - pivot_weighted_stat[quant_dlev]
        time_series_weighted_diff = pivot_weighted_stat[['year_month', quant_intan_col, 'ret_diff']]

        weighted_std_errors = []
        weighted_confidence_intervals = []
        weighted_p_values = []
        weighted_t_stats = []

        # Loop through first and last quant_intan values and perform the regression
        for quant_intan_value in [time_series_weighted_diff[quant_intan_col].min(), time_series_weighted_diff[quant_intan_col].max()]:
            df_filtered_weighted = time_series_weighted_diff[time_series_weighted_diff[quant_intan_col] == quant_intan_value]
            model_weighted = regress_on_constant(df_filtered_weighted)
            weighted_t_stats.append(model_weighted.tvalues[0])
            weighted_std_errors.append(model_weighted.bse[0]*100)
            weighted_ci = model_weighted.conf_int().iloc[0]*100
            weighted_confidence_intervals.append((weighted_ci[0], weighted_ci[1]))
            weighted_p_values.append(model_weighted.pvalues[0])

        # weighted_t_stats_row = pd.DataFrame([weighted_t_stats], columns=time_series_weighted_diff[quant_intan_col].unique())
        weighted_t_stats_row = pd.DataFrame([weighted_t_stats], columns=[time_series_weighted_diff[quant_intan_col].min(), time_series_weighted_diff[quant_intan_col].max()])
        weighted_std_errors_row = pd.DataFrame([weighted_std_errors], columns=[time_series_weighted_diff[quant_intan_col].min(), time_series_weighted_diff[quant_intan_col].max()])

        pivot_weighted_df_percent = pivot_weighted_df * 100
        pivot_weighted_df_with_t_stats = pivot_weighted_df_percent._append(weighted_t_stats_row, ignore_index=True)

        # Extract the last row (t-stats) and the rows with coefficients
        t_stats_row = pivot_df_with_t_stats.iloc[-1]
        avr_rows = pivot_df_with_t_stats.iloc[:-1]
        weighted_t_stats_row = pivot_weighted_df_with_t_stats.iloc[-1]
        weighted_avr_rows = pivot_weighted_df_with_t_stats.iloc[:-1]
        
        # Extract the rows with coefficients
        avr_rows = pivot_df_percent
        weighted_avr_rows = pivot_weighted_df_percent

        # Add stars based on p-values
        std_errors_with_stars = [f"({se:.2f}){add_stars(pval)}" for se, pval in zip(std_errors, p_values)]
        weighted_std_errors_with_stars = [f"({se:.2f}){add_stars(pval)}" for se, pval in zip(weighted_std_errors, weighted_p_values)]
        ci_strings = [f"[{ci[0]:.2f}, {ci[1]:.2f}]" for ci in confidence_intervals]
        weighted_ci_strings = [f"[{ci[0]:.2f}, {ci[1]:.2f}]" for ci in weighted_confidence_intervals]

        # Add stars to t-stats
        # t_stats_with_stars = [f"({t_stat:.2f}){add_stars(t_stat)}" for t_stat in t_stats_row]
        # weighted_t_stats_with_stars = [f"({t_stat:.2f}){add_stars(t_stat)}" for t_stat in weighted_t_stats_row]
        
        t_stats_with_stars = [f"({t_stat:.2f}){add_stars(t_stat)}" for t_stat in t_stats]
        weighted_t_stats_with_stars = [f"({t_stat:.2f}){add_stars(t_stat)}" for t_stat in weighted_t_stats]
        
        # Create the rows for the table
        for idx, row in avr_rows.iterrows():
            quant_dlev_rows = row.name
            eqw_coeffs = [f"{row[col]:.2f}" for col in [avr_rows.columns[0], avr_rows.columns[-1]]]
            latex_output += f"{quant_dlev_rows} & " + " & ".join(eqw_coeffs) + " & & "

            # Data rows for weighted average
            wgt_row = weighted_avr_rows.loc[quant_dlev_rows]
            wgt_coeffs = [f"{wgt_row[col]:.2f}" for col in [weighted_avr_rows.columns[0], weighted_avr_rows.columns[-1]]]
            latex_output += " & ".join(wgt_coeffs) + " \\\\\n"
        
        latex_output += "\\midrule\n"
        # T-statistics row
        eqw_t_stats_row = "(t-statistics) & " + " & ".join(t_stats_with_stars) + " & & "
        wgt_t_stats_row = " & ".join(weighted_t_stats_with_stars) + " \\\\\n"
        latex_output += eqw_t_stats_row + wgt_t_stats_row
        
        # Standard errors row
        eqw_se_row = "(Std. Errors) & " + " & ".join(std_errors_with_stars) + " & & "
        wgt_se_row = " & ".join(weighted_std_errors_with_stars) + " \\\\\n"
        latex_output += eqw_se_row + wgt_se_row

        # Confidence intervals row
        eqw_ci_row = "(95\% CI) & " + " & ".join(ci_strings) + " & & "
        wgt_ci_row = " & ".join(weighted_ci_strings) + " \\\\\n"
        latex_output += eqw_ci_row + wgt_ci_row

        # latex_output += "\\midrule\n"
        if col == 'ret_lead1':
            latex_output += "\midrule \n"
            latex_output += "Panel B: 2-months ahead average returns & & & & & \\\\ \n"
            latex_output += "\midrule\n"

    latex_output += """
\\bottomrule
\\end{tabular*}
\\caption{}
\\label{table:comparison}
\\end{table}
"""
    
    create_latex_table_file(latex_output, filename)
    
# def create_dlev_intan_port(df, quant_dlev, quant_intan, quant_pd, filename):
#     df_copy = df.copy()
#     quant_dlev_col = f'dlev_{quant_dlev}'
#     quant_intan_col = f'intan_at_{quant_intan}'
#     quant_pd_col = f'pd_{quant_pd}'
    
#     #df_subsample = df_copy[df_copy[quant_pd_col] == 1]
#     df_subsample = df_copy

#     grouped = df_subsample.groupby(['year_month', quant_intan_col, quant_dlev_col])

#     # Count the number of stocks in each group
#     group_counts = grouped.size().reset_index(name='counts')

#     # Filter out the groups that have fewer than 30 stocks
#     filtered_groups = group_counts[group_counts['counts'] >= 30]

#     # Merge the filtered groups back to the original DataFrame to keep only the valid groups
#     filtered_df = df_subsample.merge(filtered_groups.drop('counts', axis=1), 
#                                 on=['year_month', quant_intan_col, quant_dlev_col], 
#                                 how='inner')

#     # Re-group if necessary for further processing
#     filtered_grouped = filtered_df.groupby(['year_month', quant_intan_col, quant_dlev_col])

#     columns_to_process = ['ret_lead1', 'ret_lead2']
    
#     latex_output = """
    
# \\begin{table}    
# \\centering
# \\small
# \\setlength{\\tabcolsep}{3pt} % Adjust horizontal spacing
# \\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l*{2}{c}l*{2}{c}}
# \\toprule
# \\multicolumn{5}{c}{Intangible/assets portfolio} \\\\
# \\cmidrule(r){2-3} \\cmidrule(l){4-5}
# Leverage & \\multicolumn{2}{c}{Equal-weighted} & & \\multicolumn{2}{c}{Value-weighted} \\\\
# \\cmidrule(r){2-3} \\cmidrule(l){4-5}
# change portfolio & 1 (Lowest) & 3 (Highest) & & 1 (Lowest) & 3 (Highest) \\\\
# \\midrule
# """

#     for idx, col in enumerate(columns_to_process):
#         # Equal-weighted average returns
#         monthly_avg_returns = filtered_grouped[col].mean().reset_index()

#         # Weighted average returns
#         monthly_weighted_avg_returns = filtered_grouped.apply(lambda x: (x[col] * x['me']).sum() / x['me'].sum()).reset_index(name=f'weighted_{col}')
        
#         # Compute the Average Across All Months for equal-weighted
#         overall_avg_returns = monthly_avg_returns.groupby([quant_intan_col, quant_dlev_col])[col].mean().reset_index()

#         # Compute the Average Across All Months for weighted average
#         overall_weighted_avg_returns = monthly_weighted_avg_returns.groupby([quant_intan_col, quant_dlev_col])[f'weighted_{col}'].mean().reset_index()

#         # Pivot the DataFrame to get the correct format for LaTeX
#         pivot_df = overall_avg_returns.pivot(index=quant_dlev_col, columns=quant_intan_col, values=col)
#         pivot_weighted_df = overall_weighted_avg_returns.pivot(index=quant_dlev_col, columns=quant_intan_col, values=f'weighted_{col}')

#         # Calculate the difference between the first and last rows
#         difference_row = pivot_df.iloc[0] - pivot_df.iloc[-1]
#         difference_row.name = f'1-{quant_dlev} Difference'

#         weighted_difference_row = pivot_weighted_df.iloc[0] - pivot_weighted_df.iloc[-1]
#         weighted_difference_row.name = f'1-{quant_dlev} Difference'

#         # Append the difference row to the pivoted DataFrame
#         pivot_df = pivot_df._append(difference_row)
#         pivot_weighted_df = pivot_weighted_df._append(weighted_difference_row)

#         ######################
#         # Calculating t-stats (Equal-weighted)
#         ######################

#         pivot_t_stat = monthly_avg_returns.pivot_table(values=col, index=['year_month', quant_intan_col], columns=quant_dlev_col).reset_index()
#         pivot_t_stat['ret_diff'] = pivot_t_stat[1] - pivot_t_stat[quant_dlev]
#         time_series_diff = pivot_t_stat[['year_month', quant_intan_col, 'ret_diff']]

#         # Dictionary to store regression results
#         # t_stats = []
#         std_errors = []
#         confidence_intervals = []

#         # Loop through first and last quant_intan values and perform the regression
#         for quant_intan_value in [time_series_diff[quant_intan_col].min(), time_series_diff[quant_intan_col].max()]:
#             df_filtered = time_series_diff[time_series_diff[quant_intan_col] == quant_intan_value]
#             model = regress_on_constant(df_filtered)
#             # t_stats.append(model.tvalues[0])
#             std_errors.append(model.bse[0])
#             ci = model.conf_int().iloc[0]
#             confidence_intervals.append((ci[0], ci[1]))

#         # t_stats_row = pd.DataFrame([t_stats], columns=[time_series_diff[quant_intan_col].min(), time_series_diff[quant_intan_col].max()])
#         std_errors_row = pd.DataFrame([std_errors], columns=[time_series_diff[quant_intan_col].min(), time_series_diff[quant_intan_col].max()])

#         pivot_df_percent = pivot_df * 100
#         # pivot_df_with_t_stats = pivot_df_percent._append(t_stats_row, ignore_index=True)

#         ######################
#         # Calculating t-stats (Weighted)
#         ######################

#         pivot_weighted_t_stat = monthly_weighted_avg_returns.pivot_table(values=f'weighted_{col}', index=['year_month', quant_intan_col], columns=quant_dlev_col).reset_index()
#         pivot_weighted_t_stat['ret_diff'] = pivot_weighted_t_stat[1] - pivot_weighted_t_stat[quant_dlev]
#         time_series_weighted_diff = pivot_weighted_t_stat[['year_month', quant_intan_col, 'ret_diff']]

#         # Dictionary to store regression results
#         weighted_t_stats = []

#         # Loop through first and last quant_intan values and perform the regression
#         for quant_intan_value in [time_series_weighted_diff[quant_intan_col].min(), time_series_weighted_diff[quant_intan_col].max()]:
#             df_filtered_weighted = time_series_weighted_diff[time_series_weighted_diff[quant_intan_col] == quant_intan_value]
#             model_weighted = regress_on_constant(df_filtered_weighted)
#             weighted_t_stats.append(model_weighted.tvalues[0]) 

#         weighted_t_stats_row = pd.DataFrame([weighted_t_stats], columns=[time_series_weighted_diff[quant_intan_col].min(), time_series_weighted_diff[quant_intan_col].max()])

#         pivot_weighted_df_percent = pivot_weighted_df * 100
#         pivot_weighted_df_with_t_stats = pivot_weighted_df_percent._append(weighted_t_stats_row, ignore_index=True)

#         # Extract the last row (t-stats) and the rows with coefficients
#         t_stats_row = pivot_df_with_t_stats.iloc[-1]
#         avr_rows = pivot_df_with_t_stats.iloc[:-1]
#         weighted_t_stats_row = pivot_weighted_df_with_t_stats.iloc[-1]
#         weighted_avr_rows = pivot_weighted_df_with_t_stats.iloc[:-1]

#         # Add stars to t-stats
#         # t_stats_with_stars = [f"({t_stat:.2f}){add_stars(t_stat)}" for t_stat in t_stats_row]
#         # weighted_t_stats_with_stars = [f"({t_stat:.2f}){add_stars(t_stat)}" for t_stat in weighted_t_stats_row]
        
#         std_errors_with_stars = [f"({se:.2f}){add_stars(se)}" for se in std_errors]
#         ci_strings = [f"[{ci[0]:.2f}, {ci[1]:.2f}]" for ci in confidence_intervals]

#         # Create the rows for the table
#         for idx, row in avr_rows.iterrows():
#             quant_dlev_rows = row.name
#             eqw_coeffs = [f"{row[col]:.2f}" for col in [avr_rows.columns[0], avr_rows.columns[-1]]]
#             latex_output += f"{quant_dlev_rows} & " + " & ".join(eqw_coeffs) + " & & "

#             # Data rows for weighted average
#             wgt_row = weighted_avr_rows.loc[quant_dlev_rows]
#             wgt_coeffs = [f"{wgt_row[col]:.2f}" for col in [weighted_avr_rows.columns[0], weighted_avr_rows.columns[-1]]]
#             latex_output += " & ".join(wgt_coeffs) + " \\\\\n"

#         # Standard errors row
#         eqw_se_row = "(Std. Errors) & " + " & ".join(std_errors_with_stars[:2]) + " & & "
#         wgt_se_row = " & ".join(std_errors_with_stars[2:]) + " \\\\\n"
#         latex_output += eqw_se_row + wgt_se_row

#         # Confidence intervals row
#         eqw_ci_row = "(95% CI) & " + " & ".join(ci_strings[:2]) + " & & "
#         wgt_ci_row = " & ".join(ci_strings[2:]) + " \\\\\n"
#         latex_output += eqw_ci_row + wgt_ci_row

#         # # T-statistics row
#         # eqw_t_stats_row = "(t-statistics) & " + " & ".join([t_stats_with_stars[0], t_stats_with_stars[-1]]) + " & & "
#         # wgt_t_stats_row = " & ".join([weighted_t_stats_with_stars[0], weighted_t_stats_with_stars[-1]]) + " \\\\\n"
#         # latex_output += eqw_t_stats_row + wgt_t_stats_row

#         latex_output += "\\midrule\n"

#     latex_output += """
#     \\bottomrule
#     \\end{tabular*}
#     \\caption{}
#     \\label{table:comparison}
#     \\end{table}
#     """
    
#     create_latex_table_file(latex_output, filename)

    
#     latex_output = """

# \\begin{table}
# \\centering
# \\small
# \\setlength{\\tabcolsep}{3pt} % Adjust horizontal spacing
# \\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l*{3}{c}l*{3}{c}}
# \\toprule
# \\multicolumn{7}{c}{Intangible/assets portfolio} \\\\
# \\cmidrule(r){2-4} \\cmidrule(l){5-7}
# Leverage & \\multicolumn{3}{c}{Equal-weighted} & & \\multicolumn{3}{c}{Value-weighted} \\\\
# \\cmidrule(r){2-4} \\cmidrule(l){5-7}
# change portfolio & 1 (Lowest) & 2 & 3 (Highest) & & 1 (Lowest) & 2 & 3 (Highest) \\\\
# \\midrule
# """

#     for idx, col in enumerate(columns_to_process):
#         # Equal-weighted average returns
#         monthly_avg_returns = filtered_grouped[col].mean().reset_index()

#         # Weighted average returns
#         monthly_weighted_avg_returns = filtered_grouped.apply(lambda x: (x[col] * x['me']).sum() / x['me'].sum()).reset_index(name=f'weighted_{col}')
        
#         # Compute the Average Across All Months for equal-weighted
#         overall_avg_returns = monthly_avg_returns.groupby([quant_intan_col, quant_dlev_col])[col].mean().reset_index()

#         # Compute the Average Across All Months for weighted average
#         overall_weighted_avg_returns = monthly_weighted_avg_returns.groupby([quant_intan_col, quant_dlev_col])[f'weighted_{col}'].mean().reset_index()

#         # Pivot the DataFrame to get the correct format for LaTeX
#         pivot_df = overall_avg_returns.pivot(index=quant_dlev_col, columns=quant_intan_col, values=col)
#         pivot_weighted_df = overall_weighted_avg_returns.pivot(index=quant_dlev_col, columns=quant_intan_col, values=f'weighted_{col}')

#         # Calculate the difference between the first and fifth rows
#         difference_row = pivot_df.iloc[0] - pivot_df.iloc[quant_dlev-1]
#         difference_row.name = f'1-{quant_dlev} Difference'

#         weighted_difference_row = pivot_weighted_df.iloc[0] - pivot_weighted_df.iloc[quant_dlev-1]
#         weighted_difference_row.name = f'1-{quant_dlev} Difference'

#         # Append the difference row to the pivoted DataFrame
#         pivot_df = pivot_df._append(difference_row)
#         pivot_weighted_df = pivot_weighted_df._append(weighted_difference_row)

#         ######################
#         # Calculating t-stats (Equal-weighted)
#         ######################

#         pivot_t_stat = monthly_avg_returns.pivot_table(values=col, index=['year_month', quant_intan_col], columns=quant_dlev_col).reset_index()
#         pivot_t_stat['ret_diff'] = pivot_t_stat[1] - pivot_t_stat[quant_dlev]
#         time_series_diff = pivot_t_stat[['year_month', quant_intan_col, 'ret_diff']]

#         # Dictionary to store regression results
#         t_stats = []

#         # Loop through each quant_intan value and perform the regression
#         for quant_intan_value in time_series_diff[quant_intan_col].unique():
#             df_filtered = time_series_diff[time_series_diff[quant_intan_col] == quant_intan_value]
#             model = regress_on_constant(df_filtered)
#             t_stats.append(model.tvalues[0]) 

#         t_stats_row = pd.DataFrame([t_stats], columns=time_series_diff[quant_intan_col].unique())

#         pivot_df_percent = pivot_df * 100
#         pivot_df_with_t_stats = pivot_df_percent._append(t_stats_row, ignore_index=True)

#         ######################
#         # Calculating t-stats (Weighted)
#         ######################

#         pivot_weighted_t_stat = monthly_weighted_avg_returns.pivot_table(values=f'weighted_{col}', index=['year_month', quant_intan_col], columns=quant_dlev_col).reset_index()
#         pivot_weighted_t_stat['ret_diff'] = pivot_weighted_t_stat[1] - pivot_weighted_t_stat[quant_dlev]
#         time_series_weighted_diff = pivot_weighted_t_stat[['year_month', quant_intan_col, 'ret_diff']]

#         # Dictionary to store regression results
#         weighted_t_stats = []

#         # Loop through each quant_intan value and perform the regression
#         for quant_intan_value in time_series_weighted_diff[quant_intan_col].unique():
#             df_filtered_weighted = time_series_weighted_diff[time_series_weighted_diff[quant_intan_col] == quant_intan_value]
#             model_weighted = regress_on_constant(df_filtered_weighted)
#             weighted_t_stats.append(model_weighted.tvalues[0]) 

#         weighted_t_stats_row = pd.DataFrame([weighted_t_stats], columns=time_series_weighted_diff[quant_intan_col].unique())

#         pivot_weighted_df_percent = pivot_weighted_df * 100
#         pivot_weighted_df_with_t_stats = pivot_weighted_df_percent._append(weighted_t_stats_row, ignore_index=True)

#         # Extract the last row (t-stats) and the rows with coefficients
#         t_stats_row = pivot_df_with_t_stats.iloc[-1]
#         avr_rows = pivot_df_with_t_stats.iloc[:-1]
#         weighted_t_stats_row = pivot_weighted_df_with_t_stats.iloc[-1]
#         weighted_avr_rows = pivot_weighted_df_with_t_stats.iloc[:-1]

#         # Add stars to t-stats
#         t_stats_with_stars = [f"({t_stat:.2f}){add_stars(t_stat)}" for t_stat in t_stats_row]
#         weighted_t_stats_with_stars = [f"({t_stat:.2f}){add_stars(t_stat)}" for t_stat in weighted_t_stats_row]

#         # Create the rows for the table
#         for idx, row in avr_rows.iterrows():
#             quant_dlev_rows = row.name
#             eqw_coeffs = [f"{row[col]:.2f}" for col in avr_rows.columns]
#             latex_output += f"{quant_dlev_rows} & " + " & ".join(eqw_coeffs) + " & & "

#             # Data rows for weighted average
#             wgt_row = weighted_avr_rows.loc[quant_dlev_rows]
#             wgt_coeffs = [f"{wgt_row[col]:.2f}" for col in weighted_avr_rows.columns]
#             latex_output += " & ".join(wgt_coeffs) + " \\\\\n"

#         # T-statistics row
#         eqw_t_stats_row = "(t-statistics) & " + " & ".join(t_stats_with_stars) + " & & "
#         wgt_t_stats_row = " & ".join(weighted_t_stats_with_stars) + " \\\\\n"
#         latex_output += eqw_t_stats_row + wgt_t_stats_row

#         latex_output += "\\midrule\n"

#     latex_output += """
# \\bottomrule
# \\end{tabular*}
# \\caption{}
# \\label{table:comparison}
# \\end{table}
# """
#     create_latex_table_file(latex_output, filename)
    


def create_dlev_pd_port(df, quant_dlev, quant_pd):
    df_copy = df.copy()
    quant_dlev_col = f'dlev_{quant_dlev}'
    quant_pd_col = f'pd_{quant_pd}'

    grouped = df_copy.groupby(['year_month', quant_pd_col, quant_dlev_col])

    # Count the number of stocks in each group
    group_counts = grouped.size().reset_index(name='counts')

    # Filter out the groups that have fewer than 30 stocks
    filtered_groups = group_counts[group_counts['counts'] >= 30]

    # Merge the filtered groups back to the original DataFrame to keep only the valid groups
    filtered_df = df_copy.merge(filtered_groups.drop('counts', axis=1), 
                                on=['year_month', quant_pd_col, quant_dlev_col], 
                                how='inner')

    # Re-group if necessary for further processing
    filtered_grouped = filtered_df.groupby(['year_month', quant_pd_col, quant_dlev_col])

    # Computing the Average Return for Each Group
    monthly_avg_returns = filtered_grouped['RET'].mean().reset_index()

    # monthly_avg_returns.head(50)
    # Compute the Average Across All Months
    overall_avg_returns = monthly_avg_returns.groupby([quant_pd_col, quant_dlev_col])['RET'].mean().reset_index()
    # avr_returns_by_qui_debt_at.head(50)
    # pivot_df = overall_avg_returns.pivot(index='qui_d_debt_at', columns='qua_intan', values='ret_3mo_lead1')
    pivot_df = overall_avg_returns.pivot(index = quant_dlev_col, columns = quant_pd_col, values='RET')

    # Calculate the difference between the first and fifth rows
    difference_row = pivot_df.iloc[0] - pivot_df.iloc[quant_dlev-1]
    difference_row.name = '1-{quant_dlev} Difference'

    # Step 7: Append the difference row to the pivoted DataFrame
    pivot_df = pivot_df._append(difference_row)

    print(pivot_df)

    ######################
    # Calculating t-stats
    ######################

    pivot_t_stat = monthly_avg_returns.pivot_table(values='RET', index=['year_month', quant_pd_col], columns = quant_dlev_col).reset_index()
    pivot_t_stat.head(50)
    pivot_t_stat['ret_diff'] = pivot_t_stat[1] - pivot_t_stat[quant_dlev]
    time_series_diff = pivot_t_stat[['year_month', quant_pd_col, 'ret_diff']]
    time_series_diff = time_series_diff.sort_values(by = [quant_pd_col, 'year_month'])

    # Dictionary to store regression results
    regression_results = {}
    t_stats = []

    # time_series_diff['qui_debt_at'].unique()
    # Loop through each qui_debt_at value and perform the regression
    for quant_pd_value in time_series_diff[quant_pd_col].unique():
        df_filtered = time_series_diff[time_series_diff[quant_pd_col] == quant_pd_value]
        model = regress_on_constant(df_filtered)
        regression_results[quant_pd_value] = model.summary()
        t_stats.append(model.tvalues[0]) 

    t_stats_row = pd.DataFrame([t_stats], columns=time_series_diff[quant_pd_col].unique())

    pivot_df_percent = pivot_df * 100
    pivot_df_with_t_stats = pivot_df_percent._append(t_stats_row, ignore_index=True)
    print(pivot_df_with_t_stats)

    ##############
    # Latex table
    ##############
    
    # Extract the last row (t-stats) and the rows with coefficients
    t_stats_row = pivot_df_with_t_stats.iloc[-1]
    avr_rows = pivot_df_with_t_stats.iloc[:-1]

    # Add stars to t-stats
    t_stats_with_stars = [f"({t_stat:.2f}){add_stars(t_stat)}" for t_stat in t_stats_row]

    # Generate LaTeX table string
    num_cols = quant_pd
    # num_rows = quant_dlev

    # Create the column specification for the tabular environment
    col_spec = "c" * (num_cols + 1)
    latex_table = f"\\begin{'tabular'}{{{col_spec}}}\n"
    latex_table += "\\toprule\n"

    # Create header row
    header = "quant\\_pd & " + " & ".join(map(str, range(1, num_cols + 1))) + " \\\\\n"
    latex_table += header
    latex_table += "\\midrule\n"

    # Create data rows
    for _, row in avr_rows.iterrows():
        quant_dlev_rows = row.name
        coeffs = [f"{row[col]:.2f}" for col in avr_rows.columns]
        latex_table += f"{quant_dlev_rows} & " + " & ".join(coeffs) + " \\\\\n"

    latex_table += "\\midrule\n"

    # Add t-statistics row (assuming `t_stats_with_stars` has the same length as the number of columns)
    t_stats_row = "(t-statistics) & " + " & ".join(t_stats_with_stars) + " \\\\\n"
    latex_table += t_stats_row

    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}"

    print(latex_table)
    
def create_dlev_lev_port(df, quant_dlev, quant_intan, quant_lev, quant_pd, subsample_intan, filename):
    df_copy = df.copy()
    quant_dlev_col = f'dlev_{quant_dlev}'
    quant_intan_col = f'intan_at_{quant_intan}'
    quant_lev_col = f'lev_{quant_lev}'
    quant_pd_col = f'pd_{quant_pd}'
    
    if subsample_intan == 'all':
        df_subsample = df_copy
    elif subsample_intan == 'hint':
        df_subsample = df_copy[df_copy[quant_intan_col] == quant_intan]
    elif subsample_intan == 'lint':
        df_subsample = df_copy[df_copy[quant_intan_col] == 1]
    else:
        raise ValueError("Invalid subsample_intan value. Choose from 'all', 'hint', and 'lint'")


    grouped = df_subsample.groupby(['year_month', quant_lev_col, quant_dlev_col])

    # Count the number of stocks in each group
    group_counts = grouped.size().reset_index(name='counts')

    # Filter out the groups that have fewer than 30 stocks
    filtered_groups = group_counts[group_counts['counts'] >= 30]

    # Merge the filtered groups back to the original DataFrame to keep only the valid groups
    filtered_df = df_subsample.merge(filtered_groups.drop('counts', axis=1), 
                                on=['year_month', quant_lev_col, quant_dlev_col], 
                                how='inner')

    # Re-group if necessary for further processing
    filtered_grouped = filtered_df.groupby(['year_month', quant_lev_col, quant_dlev_col])

    columns_to_process = ['ret_lead1', 'ret_lead2']


    latex_output = """
\\begin{table}
\\centering
\\scriptsize
\\setlength{\\tabcolsep}{3pt} % Increased horizontal spacing slightly
\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l*{8}{r}}
\\toprule
& \multicolumn{8}{c}{Leverage portfolio} \\\\
\\cmidrule{2-9}
Leverage & \multicolumn{4}{c}{Equal-weighted} & \multicolumn{4}{c}{Value-weighted} \\\\
\\cmidrule(r){2-5} \cmidrule(l){6-9}
change portfolio & 1 (Lowest) & 2 & 3 & 4 (Highest) & 1 (Lowest) & 2 & 3 & 4 (Highest) \\\\
\\midrule
\\multicolumn{9}{l}{Panel A: 1-month ahead average returns} \\\\
\\midrule
"""

    for idx, col in enumerate(columns_to_process):
        monthly_avg_returns = filtered_grouped[col].mean().reset_index()

        monthly_weighted_avg_returns = filtered_grouped.apply(lambda x: (x[col] * x['me']).sum() / x['me'].sum()).reset_index(name=f'weighted_{col}')
        
        overall_avg_returns = monthly_avg_returns.groupby([quant_lev_col, quant_dlev_col])[col].mean().reset_index()

        overall_weighted_avg_returns = monthly_weighted_avg_returns.groupby([quant_lev_col, quant_dlev_col])[f'weighted_{col}'].mean().reset_index()

        pivot_df = overall_avg_returns.pivot(index=quant_dlev_col, columns=quant_lev_col, values=col)
        pivot_weighted_df = overall_weighted_avg_returns.pivot(index=quant_dlev_col, columns=quant_lev_col, values=f'weighted_{col}')

        difference_row = pivot_df.iloc[0] - pivot_df.iloc[quant_dlev-1]
        difference_row.name = f'1-{quant_dlev} Difference'

        weighted_difference_row = pivot_weighted_df.iloc[0] - pivot_weighted_df.iloc[quant_dlev-1]
        weighted_difference_row.name = f'1-{quant_dlev} Difference'

        pivot_df = pivot_df._append(difference_row)
        pivot_weighted_df = pivot_weighted_df._append(weighted_difference_row)

        ######################
        # Calculating std errors and CIs (Equal-weighted)
        ######################

        pivot_stat = monthly_avg_returns.pivot_table(values=col, index=['year_month', quant_lev_col], columns=quant_dlev_col).reset_index()
        pivot_stat['ret_diff'] = pivot_stat[1] - pivot_stat[quant_dlev]
        time_series_diff = pivot_stat[['year_month', quant_lev_col, 'ret_diff']]

        std_errors = []
        confidence_intervals = []
        p_values = []
        
        pivot_t_stat = monthly_avg_returns.pivot_table(values=col, index=['year_month', quant_lev_col], columns=quant_dlev_col).reset_index()
        pivot_t_stat['ret_diff'] = pivot_t_stat[1] - pivot_t_stat[quant_dlev]
        time_series_diff = pivot_t_stat[['year_month', quant_lev_col, 'ret_diff']]

        t_stats = []

        sorted_quant_lev_values = sorted(time_series_diff[quant_lev_col].unique())

        for quant_lev_value in sorted_quant_lev_values:
            df_filtered = time_series_diff[time_series_diff[quant_lev_col] == quant_lev_value]
            model = regress_on_constant(df_filtered)
            t_stats.append(model.tvalues[0]) 
            std_errors.append(model.bse[0]*100)
            ci = model.conf_int().iloc[0]*100
            confidence_intervals.append((ci[0], ci[1]))
            p_values.append(model.pvalues[0])
            
        t_stats_row = pd.DataFrame([t_stats], columns=time_series_diff[quant_lev_col].unique())

        pivot_df_percent = pivot_df * 100
        pivot_df_with_t_stats = pivot_df_percent._append(t_stats_row, ignore_index=True)
        # pivot_df_with_t_stats = pivot_df_percent._append(t_stats_row)

        ######################
        # Calculating std errors and CIs (Weighted)
        ######################

        pivot_weighted_stat = monthly_weighted_avg_returns.pivot_table(values=f'weighted_{col}', index=['year_month', quant_lev_col], columns=quant_dlev_col).reset_index()
        pivot_weighted_stat['ret_diff'] = pivot_weighted_stat[1] - pivot_weighted_stat[quant_dlev]
        time_series_weighted_diff = pivot_weighted_stat[['year_month', quant_lev_col, 'ret_diff']]

        weighted_std_errors = []
        weighted_confidence_intervals = []
        weighted_p_values = []
        weighted_t_stats = []
        
        pivot_weighted_t_stat = monthly_weighted_avg_returns.pivot_table(values=f'weighted_{col}', index=['year_month', quant_lev_col], columns=quant_dlev_col).reset_index()
        pivot_weighted_t_stat['ret_diff'] = pivot_weighted_t_stat[1] - pivot_weighted_t_stat[quant_dlev]
        time_series_weighted_diff = pivot_weighted_t_stat[['year_month', quant_lev_col, 'ret_diff']]

        sorted_quant_lev_values_weighted = sorted(time_series_weighted_diff[quant_lev_col].unique())

        for quant_lev_value in sorted_quant_lev_values_weighted:
            df_filtered_weighted = time_series_weighted_diff[time_series_weighted_diff[quant_lev_col] == quant_lev_value]
            model_weighted = regress_on_constant(df_filtered_weighted)
            weighted_t_stats.append(model_weighted.tvalues[0]) 
            weighted_std_errors.append(model_weighted.bse[0]*100)
            weighted_ci = model_weighted.conf_int().iloc[0]*100
            weighted_confidence_intervals.append((weighted_ci[0], weighted_ci[1]))
            weighted_p_values.append(model_weighted.pvalues[0])
            
        weighted_t_stats_row = pd.DataFrame([weighted_t_stats], columns=time_series_weighted_diff[quant_lev_col].unique())

        pivot_weighted_df_percent = pivot_weighted_df * 100
        pivot_weighted_df_with_t_stats = pivot_weighted_df_percent._append(weighted_t_stats_row, ignore_index=True)

        t_stats_row = pivot_df_with_t_stats.iloc[-1]
        avr_rows = pivot_df_with_t_stats.iloc[:-1]
        weighted_t_stats_row = pivot_weighted_df_with_t_stats.iloc[-1]
        weighted_avr_rows = pivot_weighted_df_with_t_stats.iloc[:-1]

        # Extract the rows with coefficients
        avr_rows = pivot_df_percent
        weighted_avr_rows = pivot_weighted_df_percent
        
        # Add stars based on p-values
        std_errors_with_stars = [f"({se:.2f}){add_stars(pval)}" for se, pval in zip(std_errors, p_values)]
        weighted_std_errors_with_stars = [f"({se:.2f}){add_stars(pval)}" for se, pval in zip(weighted_std_errors, weighted_p_values)]
        ci_strings = [f"[{ci[0]:.2f}, {ci[1]:.2f}]" for ci in confidence_intervals]
        weighted_ci_strings = [f"[{ci[0]:.2f}, {ci[1]:.2f}]" for ci in weighted_confidence_intervals]

        t_stats_with_stars = [f"({t_stat:.2f}){add_stars(t_stat)}" for t_stat in t_stats_row]
        weighted_t_stats_with_stars = [f"({t_stat:.2f}){add_stars(t_stat)}" for t_stat in weighted_t_stats_row]

        for idx, row in avr_rows.iterrows():
            quant_dlev_rows = row.name
            eqw_coeffs = [f"{row[col]:.2f}" for col in avr_rows.columns]
            latex_output += f"{quant_dlev_rows} & " + " & ".join(eqw_coeffs) + " & "

            wgt_row = weighted_avr_rows.loc[quant_dlev_rows]
            wgt_coeffs = [f"{wgt_row[col]:.2f}" for col in weighted_avr_rows.columns]
            latex_output += " & ".join(wgt_coeffs) + " \\\\\n"
            
        latex_output += "\\midrule\n"
        # T-statistics row
        eqw_t_stats_row = "(t-statistics) & " + " & ".join(t_stats_with_stars) + " & "
        wgt_t_stats_row = " & ".join(weighted_t_stats_with_stars) + " \\\\\n"
        latex_output += eqw_t_stats_row + wgt_t_stats_row

        # Standard errors row
        eqw_se_row = "(Std. Errors) & " + " & ".join(std_errors_with_stars) + " & "
        wgt_se_row = " & ".join(weighted_std_errors_with_stars) + " \\\\\n"
        latex_output += eqw_se_row + wgt_se_row

        # Confidence intervals row
        eqw_ci_row = "(95\% CI) & " + " & ".join(ci_strings) + " & "
        wgt_ci_row = " & ".join(weighted_ci_strings) + " \\\\\n"
        latex_output += eqw_ci_row + wgt_ci_row
        
        # latex_output += "\\midrule\n"
        if col == 'ret_lead1':
            latex_output += "\midrule \n"
            latex_output += "\multicolumn{9}{l}{Panel B: 2-months ahead average returns} \\\\ \n"
            latex_output += "\midrule\n"
            
    latex_output += """
\\bottomrule
\\end{tabular*}
\\caption{}
\\label{table:comparison}
\\end{table}
"""
    create_latex_table_file(latex_output, filename)
    

#     for idx, col in enumerate(columns_to_process):
#         # Equal-weighted average returns
#         monthly_avg_returns = filtered_grouped[col].mean().reset_index()

#         # Weighted average returns
#         monthly_weighted_avg_returns = filtered_grouped.apply(lambda x: (x[col] * x['me']).sum() / x['me'].sum()).reset_index(name=f'weighted_{col}')
        
#         # Compute the Average Across All Months for equal-weighted
#         overall_avg_returns = monthly_avg_returns.groupby([quant_intan_col, quant_dlev_col])[col].mean().reset_index()

#         # Compute the Average Across All Months for weighted average
#         overall_weighted_avg_returns = monthly_weighted_avg_returns.groupby([quant_intan_col, quant_dlev_col])[f'weighted_{col}'].mean().reset_index()

#         # Pivot the DataFrame to get the correct format for LaTeX
#         pivot_df = overall_avg_returns.pivot(index=quant_dlev_col, columns=quant_intan_col, values=col)
#         pivot_weighted_df = overall_weighted_avg_returns.pivot(index=quant_dlev_col, columns=quant_intan_col, values=f'weighted_{col}')

#         # Calculate the difference between the first and fifth rows
#         difference_row = pivot_df.iloc[0] - pivot_df.iloc[quant_dlev-1]
#         difference_row.name = f'1-{quant_dlev} Difference'

#         weighted_difference_row = pivot_weighted_df.iloc[0] - pivot_weighted_df.iloc[quant_dlev-1]
#         weighted_difference_row.name = f'1-{quant_dlev} Difference'

#         # Append the difference row to the pivoted DataFrame
#         pivot_df = pivot_df._append(difference_row)
#         pivot_weighted_df = pivot_weighted_df._append(weighted_difference_row)

#         ######################
#         # Calculating t-stats (Equal-weighted)
#         ######################

#         pivot_t_stat = monthly_avg_returns.pivot_table(values=col, index=['year_month', quant_intan_col], columns=quant_dlev_col).reset_index()
#         pivot_t_stat['ret_diff'] = pivot_t_stat[1] - pivot_t_stat[quant_dlev]
#         time_series_diff = pivot_t_stat[['year_month', quant_intan_col, 'ret_diff']]

#         # Dictionary to store regression results
#         t_stats = []

#         # Loop through each quant_intan value and perform the regression
#         for quant_intan_value in time_series_diff[quant_intan_col].unique():
#             df_filtered = time_series_diff[time_series_diff[quant_intan_col] == quant_intan_value]
#             model = regress_on_constant(df_filtered)
#             t_stats.append(model.tvalues[0]) 

#         t_stats_row = pd.DataFrame([t_stats], columns=time_series_diff[quant_intan_col].unique())

#         pivot_df_percent = pivot_df * 100
#         pivot_df_with_t_stats = pivot_df_percent._append(t_stats_row, ignore_index=True)

#         ######################
#         # Calculating t-stats (Weighted)
#         ######################

#         pivot_weighted_t_stat = monthly_weighted_avg_returns.pivot_table(values=f'weighted_{col}', index=['year_month', quant_intan_col], columns=quant_dlev_col).reset_index()
#         pivot_weighted_t_stat['ret_diff'] = pivot_weighted_t_stat[1] - pivot_weighted_t_stat[quant_dlev]
#         time_series_weighted_diff = pivot_weighted_t_stat[['year_month', quant_intan_col, 'ret_diff']]

#         # Dictionary to store regression results
#         weighted_t_stats = []

#         # Loop through each quant_intan value and perform the regression
#         for quant_intan_value in time_series_weighted_diff[quant_intan_col].unique():
#             df_filtered_weighted = time_series_weighted_diff[time_series_weighted_diff[quant_intan_col] == quant_intan_value]
#             model_weighted = regress_on_constant(df_filtered_weighted)
#             weighted_t_stats.append(model_weighted.tvalues[0]) 

#         weighted_t_stats_row = pd.DataFrame([weighted_t_stats], columns=time_series_weighted_diff[quant_intan_col].unique())

#         pivot_weighted_df_percent = pivot_weighted_df * 100
#         pivot_weighted_df_with_t_stats = pivot_weighted_df_percent._append(weighted_t_stats_row, ignore_index=True)

#         # Extract the last row (t-stats) and the rows with coefficients
#         t_stats_row = pivot_df_with_t_stats.iloc[-1]
#         avr_rows = pivot_df_with_t_stats.iloc[:-1]
#         weighted_t_stats_row = pivot_weighted_df_with_t_stats.iloc[-1]
#         weighted_avr_rows = pivot_weighted_df_with_t_stats.iloc[:-1]

#         # Add stars to t-stats
#         t_stats_with_stars = [f"({t_stat:.2f}){add_stars(t_stat)}" for t_stat in t_stats_row]
#         weighted_t_stats_with_stars = [f"({t_stat:.2f}){add_stars(t_stat)}" for t_stat in weighted_t_stats_row]

#         # Create the rows for the table
#         for idx, row in avr_rows.iterrows():
#             quant_dlev_rows = row.name
#             eqw_coeffs = [f"{row[col]:.2f}" for col in avr_rows.columns]
#             latex_output += f"{quant_dlev_rows} & " + " & ".join(eqw_coeffs) + " & & "

#             # Data rows for weighted average
#             wgt_row = weighted_avr_rows.loc[quant_dlev_rows]
#             wgt_coeffs = [f"{wgt_row[col]:.2f}" for col in weighted_avr_rows.columns]
#             latex_output += " & ".join(wgt_coeffs) + " \\\\\n"

#         # T-statistics row
#         eqw_t_stats_row = "(t-statistics) & " + " & ".join(t_stats_with_stars) + " & & "
#         wgt_t_stats_row = " & ".join(weighted_t_stats_with_stars) + " \\\\\n"
#         latex_output += eqw_t_stats_row + wgt_t_stats_row

#         latex_output += "\\midrule\n"

#     latex_output += """
# \\bottomrule
# \\end{tabular*}
# \\caption{}
# \\label{table:comparison}
# \\end{table}
# """
#     create_latex_table_file(latex_output, filename)
    
    
# def create_dlev_lev_port(df, quant_dlev, quant_intan, quant_lev, intan_subsample):
#     quant_dlev_col = f'dlev_{quant_dlev}'
#     quant_intan_col = f'intan_at_{quant_intan}'
#     quant_lev_col = f'lev_{quant_lev}'
#     df['ret_lead3'] = df.groupby('GVKEY')['RET'].shift(-1)

#     df_subsample = df[df[quant_intan_col] == intan_subsample]
#     # df_subsample = df

#     # df.shape
#     # df_subsample.shape
#     grouped = df_subsample.groupby(['year_month', quant_lev_col, quant_dlev_col])
#     # grouped.head(50)

#     # Count the number of stocks in each group
#     group_counts = grouped.size().reset_index(name='counts')

#     # Filter out the groups that have fewer than 30 stocks
#     filtered_groups = group_counts[group_counts['counts'] >= 30]

#     # Merge the filtered groups back to the original DataFrame to keep only the valid groups
#     filtered_df = df_subsample.merge(filtered_groups.drop('counts', axis=1), 
#                                 on=['year_month', quant_lev_col, quant_dlev_col], 
#                                 how='inner')

#     # Re-group if necessary for further processing
#     filtered_grouped = filtered_df.groupby(['year_month', quant_lev_col, quant_dlev_col])
    
#     # Computing the Average Return for Each Group
#     monthly_avg_returns = filtered_grouped['ret_lead3'].mean().reset_index()

#     monthly_avg_returns.head(50)
#     # Compute the Average Across All Months
#     overall_avg_returns = monthly_avg_returns.groupby([quant_lev_col, quant_dlev_col])['ret_lead3'].mean().reset_index()
#     # avr_returns_by_qui_debt_at.head(50)
#     # pivot_df = overall_avg_returns.pivot(index='qui_d_debt_at', columns='qua_intan', values='ret_3mo_lead1')
#     pivot_df = overall_avg_returns.pivot(index = quant_dlev_col, columns = quant_lev_col, values='ret_lead3')

#     # Calculate the difference between the first and fifth rows
#     difference_row = pivot_df.iloc[0] - pivot_df.iloc[quant_dlev - 1]
#     difference_row.name = '1-{quant_dlev} Difference'

#     # Step 7: Append the difference row to the pivoted DataFrame
#     pivot_df = pivot_df._append(difference_row)

#     print(pivot_df)

#     ######################
#     # Calculating t-stats
#     ######################

#     pivot_t_stat = monthly_avg_returns.pivot_table(values='ret_lead3', index=['year_month', quant_lev_col], columns = quant_dlev_col).reset_index()
#     pivot_t_stat.head(50)
#     pivot_t_stat['ret_diff'] = pivot_t_stat[1] - pivot_t_stat[quant_dlev]
#     time_series_diff = pivot_t_stat[['year_month', quant_lev_col, 'ret_diff']]
#     time_series_diff = time_series_diff.sort_values(by = [quant_lev_col, 'year_month'])

#     # Dictionary to store regression results
#     regression_results = {}
#     t_stats = []

#     # time_series_diff['qui_debt_at'].unique()
#     # Loop through each qui_debt_at value and perform the regression
#     for quant_lev_value in time_series_diff[quant_lev_col].unique():
#         df_filtered = time_series_diff[time_series_diff[quant_lev_col] == quant_lev_value]
#         model = regress_on_constant(df_filtered)
#         regression_results[quant_lev_value] = model.summary()
#         t_stats.append(model.tvalues[0]) 

#     t_stats_row = pd.DataFrame([t_stats], columns=time_series_diff[quant_lev_col].unique())

#     pivot_df_percent = pivot_df * 100
#     pivot_df_with_t_stats = pivot_df_percent._append(t_stats_row, ignore_index=True)
#     print(pivot_df_with_t_stats)

#     ##############
#     # Latex table
#     ##############

#     # Extract the last row (t-stats) and the rows with coefficients
#     t_stats_row = pivot_df_with_t_stats.iloc[-1]
#     avr_rows = pivot_df_with_t_stats.iloc[:-1]

#     # Add stars to t-stats
#     t_stats_with_stars = [f"({t_stat:.2f}){add_stars(t_stat)}" for t_stat in t_stats_row]

#     # Generate LaTeX table string
#     num_cols = quant_lev
#     # num_rows = quant_dlev

#     # Create the column specification for the tabular environment
#     col_spec = "c" * (num_cols + 1)
#     latex_table = f"\\begin{'tabular'}{{{col_spec}}}\n"
#     latex_table += "\\toprule\n"

#     # Create header row
#     header = "quant\\_lev & " + " & ".join(map(str, range(1, num_cols + 1))) + " \\\\\n"
#     latex_table += header
#     latex_table += "\\midrule\n"

#     # Create data rows
#     for _, row in avr_rows.iterrows():
#         quant_dlev_rows = row.name
#         coeffs = [f"{row[col]:.2f}" for col in avr_rows.columns]
#         latex_table += f"{quant_dlev_rows} & " + " & ".join(coeffs) + " \\\\\n"

#     latex_table += "\\midrule\n"

#     # Add t-statistics row (assuming `t_stats_with_stars` has the same length as the number of columns)
#     t_stats_row = "(t-statistics) & " + " & ".join(t_stats_with_stars) + " \\\\\n"
#     latex_table += t_stats_row

#     latex_table += "\\bottomrule\n"
#     latex_table += "\\end{tabular}"

#     print(latex_table)
    
    
    
def create_dlev_lev_intan(df, quant_dlev, quant_lev):
    quant_dlev_col = f'dlev_{quant_dlev}'
    # quant_intan_col = f'intan_at_{quant_intan}'
    quant_lev_col = f'lev_{quant_lev}'

    # df_subsample = df[df[quant_intan_col] == intan_subsample]
    # df_subsample = df

    # df.shape
    # df_subsample.shape
    grouped = df.groupby(['year_month', quant_lev_col, quant_dlev_col])
    # grouped.head(50)

    # Count the number of stocks in each group
    group_counts = grouped.size().reset_index(name='counts')

    # Filter out the groups that have fewer than 30 stocks
    filtered_groups = group_counts[group_counts['counts'] >= 30]

    # Merge the filtered groups back to the original DataFrame to keep only the valid groups
    filtered_df = df.merge(filtered_groups.drop('counts', axis=1), 
                                on=['year_month', quant_lev_col, quant_dlev_col], 
                                how='inner')

    # Re-group if necessary for further processing
    filtered_grouped = filtered_df.groupby(['year_month', quant_lev_col, quant_dlev_col])
    
    # Computing the Average Return for Each Group
    # monthly_avg_returns = filtered_grouped['intan_epk_at'].mean().reset_index()
    weighted_avg_returns = filtered_grouped.apply(
        lambda x: np.average(x['intan_epk_at'], weights=x['atq'])
        ).reset_index(name='intan_epk_at')

    # Compute the Average Across All Months
    # monthly_avg_returns
    overall_avg_returns = weighted_avg_returns.groupby([quant_lev_col, quant_dlev_col])['intan_epk_at'].mean().reset_index()
    # avr_returns_by_qui_debt_at.head(50)
    # pivot_df = overall_avg_returns.pivot(index='qui_d_debt_at', columns='qua_intan', values='ret_3mo_lead1')
    pivot_df = overall_avg_returns.pivot(index = quant_dlev_col, columns = quant_lev_col, values='intan_epk_at')

    # Calculate the difference between the first and fifth rows
    difference_row = pivot_df.iloc[0] - pivot_df.iloc[quant_dlev - 1]
    difference_row.name = '1-{quant_dlev} Difference'

    # Step 7: Append the difference row to the pivoted DataFrame
    pivot_df = pivot_df._append(difference_row)

    print(pivot_df)

    ######################
    # Calculating t-stats
    ######################

    pivot_t_stat = weighted_avg_returns.pivot_table(values='intan_epk_at', index=['year_month', quant_lev_col], columns = quant_dlev_col).reset_index()
    pivot_t_stat.head(50)
    pivot_t_stat['ret_diff'] = pivot_t_stat[1] - pivot_t_stat[quant_dlev]
    time_series_diff = pivot_t_stat[['year_month', quant_lev_col, 'ret_diff']]
    time_series_diff = time_series_diff.sort_values(by = [quant_lev_col, 'year_month'])

    # Dictionary to store regression results
    regression_results = {}
    t_stats = []

    # time_series_diff['qui_debt_at'].unique()
    # Loop through each qui_debt_at value and perform the regression
    for quant_lev_value in time_series_diff[quant_lev_col].unique():
        df_filtered = time_series_diff[time_series_diff[quant_lev_col] == quant_lev_value]
        model = regress_on_constant(df_filtered)
        regression_results[quant_lev_value] = model.summary()
        t_stats.append(model.tvalues[0]) 

    t_stats_row = pd.DataFrame([t_stats], columns=time_series_diff[quant_lev_col].unique())

    pivot_df_percent = pivot_df
    pivot_df_with_t_stats = pivot_df_percent._append(t_stats_row, ignore_index=True)
    print(pivot_df_with_t_stats)

    ##############
    # Latex table
    ##############

    # Extract the last row (t-stats) and the rows with coefficients
    t_stats_row = pivot_df_with_t_stats.iloc[-1]
    avr_rows = pivot_df_with_t_stats.iloc[:-1]

    # Add stars to t-stats
    t_stats_with_stars = [f"({t_stat:.2f}){add_stars(t_stat)}" for t_stat in t_stats_row]

    # Generate LaTeX table string
    num_cols = quant_lev
    # num_rows = quant_dlev

    # Create the column specification for the tabular environment
    col_spec = "c" * (num_cols + 1)
    latex_table = f"\\begin{'tabular'}{{{col_spec}}}\n"
    latex_table += "\\toprule\n"

    # Create header row
    header = "quant\\_lev & " + " & ".join(map(str, range(1, num_cols + 1))) + " \\\\\n"
    latex_table += header
    latex_table += "\\midrule\n"

    # Create data rows
    for _, row in avr_rows.iterrows():
        quant_dlev_rows = row.name
        coeffs = [f"{row[col]:.2f}" for col in avr_rows.columns]
        latex_table += f"{quant_dlev_rows} & " + " & ".join(coeffs) + " \\\\\n"

    latex_table += "\\midrule\n"

    # Add t-statistics row (assuming `t_stats_with_stars` has the same length as the number of columns)
    t_stats_row = "(t-statistics) & " + " & ".join(t_stats_with_stars) + " \\\\\n"
    latex_table += t_stats_row

    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}"

    print(latex_table)
    
def create_lev_pd_intan(df, quant_lev):
    quant_lev_col = f'lev_{quant_lev}'
    grouped = df.groupby(['year_month', quant_lev_col])

    # Count the number of stocks in each group
    group_counts = grouped.size().reset_index(name='counts')

    # Filter out the groups that have fewer than 30 stocks
    filtered_groups = group_counts[group_counts['counts'] >= 30]

    # Merge the filtered groups back to the original DataFrame to keep only the valid groups
    filtered_df = df.merge(filtered_groups.drop('counts', axis=1), 
                                on=['year_month', quant_lev_col], 
                                how='inner')

    # Re-group if necessary for further processing
    filtered_grouped = filtered_df.groupby(['year_month', quant_lev_col])
    
    # Computing the Average Return for Each Group
    weighted_avg_returns = filtered_grouped.apply(
        lambda x: np.average(x['intan_epk_at'], weights=x['atq'])
        ).reset_index(name='intan_epk_at')

    # Computing the Weighted Average Probability of Default for Each Group
    weighted_avg_default_prob = filtered_grouped.apply(
        lambda x: np.average(x['default_probability'], weights=x['atq'])
    ).reset_index(name='default_probability')
    
    # equal_weighted_avg_default_prob = filtered_grouped['default_probability'].mean().reset_index(name='default_probability')

    # Computing the Weighted Average Probability of Default for Each Group
    # weighted_avg_ret = filtered_grouped.apply(
    #     lambda x: np.average(x['RET'], weights=x['atq'])
    # ).reset_index(name='RET')
    
    # equal_weighted_avg_default_prob = filtered_grouped['RET'].mean().reset_index(name='RET')

    # Compute the Average Across All Months
    overall_avg_returns = weighted_avg_returns.groupby([quant_lev_col])['intan_epk_at'].mean().reset_index()
    # overall_avg_default_prob = weighted_avg_default_prob.groupby(quant_lev_col)['default_probability'].mean().reset_index()
    # overall_avg_ret = equal_weighted_avg_default_prob.groupby(quant_lev_col)['RET'].mean().reset_index()


    # final_pivot_df = overall_avg_returns.pivot(columns = quant_lev_col, values='intan_epk_at').reset_index(drop=True)
    final_df = overall_avg_returns.set_index(quant_lev_col).T
    # final_default_prob_df = overall_avg_default_prob.set_index(quant_lev_col).T
    # final_ret_df = overall_avg_ret.set_index(quant_lev_col).T

    # final_df = pd.concat([final_df, final_default_prob_df])
    final_df.index = ['Average Intangible/assets']

    print(final_df)

    ##############
    # Latex table
    ##############

    # Generate LaTeX table string
    num_cols = len(final_df.columns)
    col_spec = "c" * num_cols
    latex_table = f"\\begin{{tabular}}{{{col_spec}}}\n"
    latex_table += "\\toprule\n"

    # Create header row
    header = " & ".join(map(str, final_df.columns)) + " \\\\\n"
    latex_table += header
    latex_table += "\\midrule\n"

    # Create data rows for Average Return and Average Default Probability
    for index, row in final_df.iterrows():
        latex_table += index + " & " + " & ".join([f"{value:.2f}" for value in row]) + " \\\\\n"

    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}"

    print(latex_table)
    
    
def create_intan_lev_pd(df, quant_intan):
    quant_intan_col = f'intan_at_{quant_intan}'
    grouped = df.groupby(['year_month', quant_intan_col])

    # Count the number of stocks in each group
    group_counts = grouped.size().reset_index(name='counts')

    # Filter out the groups that have fewer than 30 stocks
    filtered_groups = group_counts[group_counts['counts'] >= 30]

    # Merge the filtered groups back to the original DataFrame to keep only the valid groups
    filtered_df = df.merge(filtered_groups.drop('counts', axis=1), 
                                on=['year_month', quant_intan_col], 
                                how='inner')

    # Re-group if necessary for further processing
    filtered_grouped = filtered_df.groupby(['year_month', quant_intan_col])
    
    # Computing the Average Return for Each Group
    weighted_avg_returns = filtered_grouped.apply(
        lambda x: np.average(x['lev'], weights=x['atq'])
        ).reset_index(name='lev')

    # Computing the Weighted Average Probability of Default for Each Group
    weighted_avg_default_prob = filtered_grouped.apply(
        lambda x: np.average(x['default_probability'], weights=x['atq'])
    ).reset_index(name='default_probability')
    
    # equal_weighted_avg_default_prob = filtered_grouped['default_probability'].mean().reset_index(name='default_probability')

    # Computing the Weighted Average Probability of Default for Each Group
    # weighted_avg_ret = filtered_grouped.apply(
    #     lambda x: np.average(x['RET'], weights=x['atq'])
    # ).reset_index(name='RET')
    
    # equal_weighted_avg_default_prob = filtered_grouped['RET'].mean().reset_index(name='RET')

    # Compute the Average Across All Months
    overall_avg_returns = weighted_avg_returns.groupby([quant_intan_col])['lev'].mean().reset_index()
    overall_avg_default_prob = weighted_avg_default_prob.groupby(quant_intan_col)['default_probability'].mean().reset_index()
    # overall_avg_ret = equal_weighted_avg_default_prob.groupby(quant_intan_col)['RET'].mean().reset_index()


    # final_pivot_df = overall_avg_returns.pivot(columns = quant_intan_col, values='intan_epk_at').reset_index(drop=True)
    final_df = overall_avg_returns.set_index(quant_intan_col).T
    final_default_prob_df = overall_avg_default_prob.set_index(quant_intan_col).T
    # final_ret_df = overall_avg_ret.set_index(quant_intan_col).T

    final_df = pd.concat([final_df, final_default_prob_df])
    final_df.index = ['Average leverage', 'Average Default Probability']

    print(final_df)

    ##############
    # Latex table
    ##############

    # Generate LaTeX table string
    num_cols = len(final_df.columns)
    col_spec = "c" * num_cols
    latex_table = f"\\begin{{tabular}}{{{col_spec}}}\n"
    latex_table += "\\toprule\n"

    # Create header row
    header = " & ".join(map(str, final_df.columns)) + " \\\\\n"
    latex_table += header
    latex_table += "\\midrule\n"

    # Create data rows for Average Return and Average Default Probability
    for index, row in final_df.iterrows():
        latex_table += index + " & " + " & ".join([f"{value:.2f}" for value in row]) + " \\\\\n"

    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}"

    print(latex_table)
    
    
    # Function to perform regression on a constant
def regress_on_constant(df):
    df = df.dropna(subset=['ret_diff'])  # Remove rows with NaN values in 'RET_diff'
    X = sm.add_constant(pd.Series([1]*len(df), index=df.index))  # Add a constant term (array of ones) with the same index as df
    model = sm.OLS(df['ret_diff'], X).fit()
    return model

def create_dlev_lev_pd(df, quant_dlev, quant_lev):
    quant_dlev_col = f'dlev_{quant_dlev}'
    # quant_intan_col = f'intan_at_{quant_intan}'
    quant_lev_col = f'lev_{quant_lev}'

    # df_subsample = df[df[quant_intan_col] == intan_subsample]
    # df_subsample = df

    # df.shape
    # df_subsample.shape
    grouped = df.groupby(['year_month', quant_lev_col, quant_dlev_col])
    # grouped.head(50)

    # Count the number of stocks in each group
    group_counts = grouped.size().reset_index(name='counts')

    # Filter out the groups that have fewer than 30 stocks
    filtered_groups = group_counts[group_counts['counts'] >= 30]

    # Merge the filtered groups back to the original DataFrame to keep only the valid groups
    filtered_df = df.merge(filtered_groups.drop('counts', axis=1), 
                                on=['year_month', quant_lev_col, quant_dlev_col], 
                                how='inner')

    # Re-group if necessary for further processing
    filtered_grouped = filtered_df.groupby(['year_month', quant_lev_col, quant_dlev_col])
    
    # Computing the Average Return for Each Group
    # monthly_avg_returns = filtered_grouped['intan_epk_at'].mean().reset_index()
    weighted_avg_returns = filtered_grouped.apply(
        lambda x: np.average(x['default_probability'], weights=x['atq'])
        ).reset_index(name='default_probability')

    # Compute the Average Across All Months
    # monthly_avg_returns
    overall_avg_returns = weighted_avg_returns.groupby([quant_lev_col, quant_dlev_col])['default_probability'].mean().reset_index()
    # avr_returns_by_qui_debt_at.head(50)
    # pivot_df = overall_avg_returns.pivot(index='qui_d_debt_at', columns='qua_intan', values='ret_3mo_lead1')
    pivot_df = overall_avg_returns.pivot(index = quant_dlev_col, columns = quant_lev_col, values='default_probability')

    # Calculate the difference between the first and fifth rows
    difference_row = pivot_df.iloc[0] - pivot_df.iloc[quant_dlev - 1]
    difference_row.name = '1-{quant_dlev} Difference'

    # Step 7: Append the difference row to the pivoted DataFrame
    pivot_df = pivot_df._append(difference_row)

    print(pivot_df)

    ######################
    # Calculating t-stats
    ######################

    pivot_t_stat = weighted_avg_returns.pivot_table(values='default_probability', index=['year_month', quant_lev_col], columns = quant_dlev_col).reset_index()
    pivot_t_stat.head(50)
    pivot_t_stat['ret_diff'] = pivot_t_stat[1] - pivot_t_stat[quant_dlev]
    time_series_diff = pivot_t_stat[['year_month', quant_lev_col, 'ret_diff']]
    time_series_diff = time_series_diff.sort_values(by = [quant_lev_col, 'year_month'])

    # Dictionary to store regression results
    regression_results = {}
    t_stats = []

    # time_series_diff['qui_debt_at'].unique()
    # Loop through each qui_debt_at value and perform the regression
    for quant_lev_value in time_series_diff[quant_lev_col].unique():
        df_filtered = time_series_diff[time_series_diff[quant_lev_col] == quant_lev_value]
        model = regress_on_constant(df_filtered)
        regression_results[quant_lev_value] = model.summary()
        t_stats.append(model.tvalues[0]) 

    t_stats_row = pd.DataFrame([t_stats], columns=time_series_diff[quant_lev_col].unique())

    pivot_df_percent = pivot_df
    pivot_df_with_t_stats = pivot_df_percent._append(t_stats_row, ignore_index=True)
    print(pivot_df_with_t_stats)

    ##############
    # Latex table
    ##############

    # Extract the last row (t-stats) and the rows with coefficients
    t_stats_row = pivot_df_with_t_stats.iloc[-1]
    avr_rows = pivot_df_with_t_stats.iloc[:-1]

    # Add stars to t-stats
    t_stats_with_stars = [f"({t_stat:.2f}){add_stars(t_stat)}" for t_stat in t_stats_row]

    # Generate LaTeX table string
    num_cols = quant_lev
    # num_rows = quant_dlev

    # Create the column specification for the tabular environment
    col_spec = "c" * (num_cols + 1)
    latex_table = f"\\begin{'tabular'}{{{col_spec}}}\n"
    latex_table += "\\toprule\n"

    # Create header row
    header = "quant\\_lev & " + " & ".join(map(str, range(1, num_cols + 1))) + " \\\\\n"
    latex_table += header
    latex_table += "\\midrule\n"

    # Create data rows
    for _, row in avr_rows.iterrows():
        quant_dlev_rows = row.name
        coeffs = [f"{row[col]:.2f}" for col in avr_rows.columns]
        latex_table += f"{quant_dlev_rows} & " + " & ".join(coeffs) + " \\\\\n"

    latex_table += "\\midrule\n"

    # Add t-statistics row (assuming `t_stats_with_stars` has the same length as the number of columns)
    t_stats_row = "(t-statistics) & " + " & ".join(t_stats_with_stars) + " \\\\\n"
    latex_table += t_stats_row

    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}"

    print(latex_table)
    

def create_dlev_pd_intan_port(df, quant_dlev, quant_intan, quant_pd, intan_subsample):
    quant_dlev_col = f'dlev_{quant_dlev}'
    quant_intan_col = f'intan_at_{quant_intan}'
    quant_pd_col = f'pd_{quant_pd}'

    df_subsample = df[df[quant_intan_col] == intan_subsample]
    # df_subsample = df

    # df.shape
    # df_subsample.shape
    grouped = df_subsample.groupby(['year_month', quant_pd_col, quant_dlev_col])
    # grouped.head(50)

    # Count the number of stocks in each group
    group_counts = grouped.size().reset_index(name='counts')

    # Filter out the groups that have fewer than 30 stocks
    filtered_groups = group_counts[group_counts['counts'] >= 30]

    # Merge the filtered groups back to the original DataFrame to keep only the valid groups
    filtered_df = df_subsample.merge(filtered_groups.drop('counts', axis=1), 
                                on=['year_month', quant_pd_col, quant_dlev_col], 
                                how='inner')

    # Re-group if necessary for further processing
    filtered_grouped = filtered_df.groupby(['year_month', quant_pd_col, quant_dlev_col])
    
    # Computing the Average Return for Each Group
    monthly_avg_returns = filtered_grouped['RET'].mean().reset_index()

    monthly_avg_returns.head(50)
    # Compute the Average Across All Months
    overall_avg_returns = monthly_avg_returns.groupby([quant_pd_col, quant_dlev_col])['RET'].mean().reset_index()
    # avr_returns_by_qui_debt_at.head(50)
    # pivot_df = overall_avg_returns.pivot(index='qui_d_debt_at', columns='qua_intan', values='ret_3mo_lead1')
    pivot_df = overall_avg_returns.pivot(index = quant_dlev_col, columns = quant_pd_col, values='RET')

    # Calculate the difference between the first and fifth rows
    difference_row = pivot_df.iloc[0] - pivot_df.iloc[quant_dlev - 1]
    difference_row.name = '1-{quant_dlev} Difference'

    # Step 7: Append the difference row to the pivoted DataFrame
    pivot_df = pivot_df._append(difference_row)

    print(pivot_df)

    ######################
    # Calculating t-stats
    ######################

    pivot_t_stat = monthly_avg_returns.pivot_table(values='RET', index=['year_month', quant_pd_col], columns = quant_dlev_col).reset_index()
    pivot_t_stat.head(50)
    pivot_t_stat['ret_diff'] = pivot_t_stat[1] - pivot_t_stat[quant_dlev]
    time_series_diff = pivot_t_stat[['year_month', quant_pd_col, 'ret_diff']]
    time_series_diff = time_series_diff.sort_values(by = [quant_pd_col, 'year_month'])

    # Dictionary to store regression results
    regression_results = {}
    t_stats = []

    # time_series_diff['qui_debt_at'].unique()
    # Loop through each qui_debt_at value and perform the regression
    for quant_pd_value in time_series_diff[quant_pd_col].unique():
        df_filtered = time_series_diff[time_series_diff[quant_pd_col] == quant_pd_value]
        model = regress_on_constant(df_filtered)
        # regression_results[quant_pd_value] = model.summary()
        t_stats.append(model.tvalues[0]) 

    t_stats_row = pd.DataFrame([t_stats], columns=time_series_diff[quant_pd_col].unique())

    pivot_df_percent = pivot_df * 100
    pivot_df_with_t_stats = pivot_df_percent._append(t_stats_row, ignore_index=True)
    print(pivot_df_with_t_stats)

    ##############
    # Latex table
    ##############

    # Extract the last row (t-stats) and the rows with coefficients
    t_stats_row = pivot_df_with_t_stats.iloc[-1]
    avr_rows = pivot_df_with_t_stats.iloc[:-1]

    # Add stars to t-stats
    t_stats_with_stars = [f"({t_stat:.2f}){add_stars(t_stat)}" for t_stat in t_stats_row]

    # Generate LaTeX table string
    num_cols = quant_pd
    # num_rows = quant_dlev

    # Create the column specification for the tabular environment
    col_spec = "c" * (num_cols + 1)
    latex_table = f"\\begin{'tabular'}{{{col_spec}}}\n"
    latex_table += "\\toprule\n"

    # Create header row
    header = "quant\\_pd & " + " & ".join(map(str, range(1, num_cols + 1))) + " \\\\\n"
    latex_table += header
    latex_table += "\\midrule\n"

    # Create data rows
    for _, row in avr_rows.iterrows():
        quant_dlev_rows = row.name
        coeffs = [f"{row[col]:.2f}" for col in avr_rows.columns]
        latex_table += f"{quant_dlev_rows} & " + " & ".join(coeffs) + " \\\\\n"

    latex_table += "\\midrule\n"

    # Add t-statistics row (assuming `t_stats_with_stars` has the same length as the number of columns)
    t_stats_row = "(t-statistics) & " + " & ".join(t_stats_with_stars) + " \\\\\n"
    latex_table += t_stats_row

    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}"

    print(latex_table)