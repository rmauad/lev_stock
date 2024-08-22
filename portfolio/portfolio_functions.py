import pandas as pd
import numpy as np
from tabulate import tabulate
from linearmodels.panel.model import FamaMacBeth
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
    
def fm_ret_lev(df):
    df_copy = df.copy()
    df_copy.reset_index(inplace=True)
    df_copy['year_month'] = df_copy['year_month'].astype(str)
    df_copy['year_month'] = pd.to_datetime(df_copy['year_month'], format='%Y-%m')
    
    # df_copy['year_month'] = df_copy['year_month'].to_timestamp()
    # df_copy['year_month'] = pd.to_datetime(df_copy['year_month'], format='%Y%m').dt.to_period('M').to_timestamp()
    # df_copy['year_month'] = pd.to_datetime(df_copy['year_month'], format='%Y%m')

    df_copy.set_index(['GVKEY', 'year_month'], inplace=True)
    
    ###################################
    # Running Fama MacBeth regressions
    ###################################
    # df_copy['quant_intan_dummy'] = df_copy['intan_at_5'] == 5
    # df_copy['dlevXintan_quant'] = df_copy['dlev'] * df_copy['quant_intan_dummy']
    # df_copy['indust_dummy'] = df_copy['ff_indust'] == 10
    # df_copy['dlevXindust_dummy'] = df_copy['dlev'] * df_copy['indust_dummy']
    
    dep_vars = ['RET_lead1', 'ret_2mo_lead1', 'ret_3mo_lead1']
    # dep = df_clean_no_na['ret_2mo_lead1']*100
    # indep_vars = ['dlev', 'dlevXindust_dummy', 'indust_dummy', 'lev', 'beta', 'ln_me', 'bm', 'roe', 'one_year_cum_return', 'RET']
    
    indep_vars = ['dlev', 'lev', 'beta', 'ln_me', 'bm', 'roe', 'one_year_cum_return', 'RET']
    
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
        'd_lev': 'Leverage Change',
        'lev': 'Leverage',
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
        df_clean_no_na = df_copy.dropna(subset=[dep_var] + indep_vars)
        dep = df_clean_no_na[dep_var] * 100
        indep = df_clean_no_na[indep_vars]
        mod = FamaMacBeth(dep, indep)
        res = mod.fit()
        results[dep_var] = res
        obs_counts[dep_var] = res.nobs
        df_reset = df_clean_no_na.reset_index()
        firm_counts[dep_var] = df_reset['GVKEY'].nunique()
        r_squared[dep_var] = res.rsquared

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

    # Add rows for number of observations, number of unique firms, and R-squared
    obs_firms_data = [
        ['Observations'] + [obs_counts[dep_var] for dep_var in dep_vars],
        ['Number of Firms'] + [firm_counts[dep_var] for dep_var in dep_vars],
        ['R-squared'] + [round(r_squared[dep_var], 5) for dep_var in dep_vars]
    ]

    # Combine all rows
    all_rows = rows + obs_firms_data

    # Generate LaTeX table
    latex_table = tabulate(all_rows, headers=['Variable'] + dep_vars, tablefmt='latex_raw', showindex=False)

    # Print LaTeX table
    print(latex_table)

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


def create_dlev_intan_port(df, quant_dlev, quant_intan, quant_pd, filename):
    df_copy = df.copy()
    quant_dlev_col = f'dlev_{quant_dlev}'
    quant_intan_col = f'intan_at_{quant_intan}'
    quant_pd_col = f'pd_{quant_pd}'
    
    df_subsample = df_copy[df_copy[quant_pd_col] == 1]
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
\\multicolumn{5}{c}{Intangible/assets portfolio} \\\\
\\cmidrule(r){2-3} \\cmidrule(l){4-5}
Leverage & \\multicolumn{2}{c}{Equal-weighted} & & \\multicolumn{2}{c}{Value-weighted} \\\\
\\cmidrule(r){2-3} \\cmidrule(l){4-5}
change portfolio & 1 (Lowest) & 3 (Highest) & & 1 (Lowest) & 3 (Highest) \\\\
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

        latex_output += "\\midrule\n"

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
    
def create_dlev_lev_port(df, quant_dlev, quant_intan, quant_lev, quant_pd, filename):
    df_copy = df.copy()
    quant_dlev_col = f'dlev_{quant_dlev}'
    quant_intan_col = f'intan_at_{quant_intan}'
    quant_lev_col = f'lev_{quant_lev}'
    quant_pd_col = f'pd_{quant_pd}'
    
    # df_subsample = df_copy[(df_copy[quant_pd_col] == quant_pd) & (df_copy[quant_intan_col] == quant_intan)]
    df_subsample = df_copy[df_copy[quant_intan_col] == 1]
    # df_subsample = df_copy

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
\\setlength{\\tabcolsep}{0.7pt} % Adjust horizontal spacing
\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l*{5}{c}l*{5}{c}}
\\toprule
\\multicolumn{7}{c}{Leverage portfolios} \\\\
\\cmidrule(r){2-6} \\cmidrule(l){7-11}
Leverage & \\multicolumn{3}{c}{Equal-weighted} & & \\multicolumn{5}{c}{Value-weighted} \\\\
\\cmidrule(r){2-6} \\cmidrule(l){7-11}
change portfolio & 1 (Lowest) & 2 & & 4 & 5 (Highest) & & 1 (Lowest) & 2 & 3 & 4 & 5 (Highest) \\\\
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

        for quant_lev_value in time_series_diff[quant_lev_col].unique():
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

        for quant_lev_value in time_series_weighted_diff[quant_lev_col].unique():
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
            latex_output += f"{quant_dlev_rows} & " + " & ".join(eqw_coeffs) + " & & "

            wgt_row = weighted_avr_rows.loc[quant_dlev_rows]
            wgt_coeffs = [f"{wgt_row[col]:.2f}" for col in weighted_avr_rows.columns]
            latex_output += " & ".join(wgt_coeffs) + " \\\\\n"

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
        
        latex_output += "\\midrule\n"

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