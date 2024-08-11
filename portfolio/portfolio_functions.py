import pandas as pd
from tabulate import tabulate
from linearmodels.panel.model import FamaMacBeth
import statsmodels.api as sm

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
    
    
def create_dlev_intan_port(df, quant_dlev, quant_intan):
    df_copy = df.copy()
    quant_dlev_col = f'dlev_{quant_dlev}'
    quant_intan_col = f'intan_at_{quant_intan}'

    grouped = df_copy.groupby(['year_month', quant_intan_col, quant_dlev_col])

    # Count the number of stocks in each group
    group_counts = grouped.size().reset_index(name='counts')

    # Filter out the groups that have fewer than 30 stocks
    filtered_groups = group_counts[group_counts['counts'] >= 30]

    # Merge the filtered groups back to the original DataFrame to keep only the valid groups
    filtered_df = df_copy.merge(filtered_groups.drop('counts', axis=1), 
                                on=['year_month', quant_intan_col, quant_dlev_col], 
                                how='inner')

    # Re-group if necessary for further processing
    filtered_grouped = filtered_df.groupby(['year_month', quant_intan_col, quant_dlev_col])

    # Computing the Average Return for Each Group
    monthly_avg_returns = filtered_grouped['RET'].mean().reset_index()

    # monthly_avg_returns.head(50)
    # Compute the Average Across All Months
    overall_avg_returns = monthly_avg_returns.groupby([quant_intan_col, quant_dlev_col])['RET'].mean().reset_index()
    # avr_returns_by_qui_debt_at.head(50)
    # pivot_df = overall_avg_returns.pivot(index='qui_d_debt_at', columns='qua_intan', values='ret_3mo_lead1')
    pivot_df = overall_avg_returns.pivot(index = quant_dlev_col, columns = quant_intan_col, values='RET')

    # Calculate the difference between the first and fifth rows
    difference_row = pivot_df.iloc[0] - pivot_df.iloc[quant_dlev-1]
    difference_row.name = '1-{quant_dlev} Difference'

    # Step 7: Append the difference row to the pivoted DataFrame
    pivot_df = pivot_df._append(difference_row)

    print(pivot_df)

    ######################
    # Calculating t-stats
    ######################

    pivot_t_stat = monthly_avg_returns.pivot_table(values='RET', index=['year_month', quant_intan_col], columns = quant_dlev_col).reset_index()
    pivot_t_stat.head(50)
    pivot_t_stat['ret_diff'] = pivot_t_stat[1] - pivot_t_stat[quant_dlev]
    time_series_diff = pivot_t_stat[['year_month', quant_intan_col, 'ret_diff']]
    time_series_diff = time_series_diff.sort_values(by = [quant_intan_col, 'year_month'])

    # Dictionary to store regression results
    regression_results = {}
    t_stats = []

    # time_series_diff['qui_debt_at'].unique()
    # Loop through each qui_debt_at value and perform the regression
    for quant_intan_value in time_series_diff[quant_intan_col].unique():
        df_filtered = time_series_diff[time_series_diff[quant_intan_col] == quant_intan_value]
        model = regress_on_constant(df_filtered)
        regression_results[quant_intan_value] = model.summary()
        t_stats.append(model.tvalues[0]) 

    t_stats_row = pd.DataFrame([t_stats], columns=time_series_diff[quant_intan_col].unique())

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
    num_cols = quant_intan
    # num_rows = quant_dlev

    # Create the column specification for the tabular environment
    col_spec = "c" * (num_cols + 1)
    latex_table = f"\\begin{'tabular'}{{{col_spec}}}\n"
    latex_table += "\\toprule\n"

    # Create header row
    header = "quant\\_intan & " + " & ".join(map(str, range(1, num_cols + 1))) + " \\\\\n"
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

    # # Extract the last row (t-stats) and the rows with coefficients
    # t_stats_row = pivot_df_with_t_stats.iloc[-1]
    # avr_rows = pivot_df_with_t_stats.iloc[:-1]

    # # Add stars to t-stats
    # t_stats_with_stars = [f"({t_stat:.2f}){add_stars(t_stat)}" for t_stat in t_stats_row]

    # # Create LaTeX table string
    # latex_table = "\\begin{tabular}{cccc}\n"
    # latex_table += "\\toprule\n"
    # latex_table += "qui\\_intan\\_at & 1 & 2 & 3 \\\\\n"
    # latex_table += "\\midrule\n"

    # for index, row in avr_rows.iterrows():
    #     qui_debt_at = row.name
    #     coeffs = [f"{row[col]:.2f}" for col in avr_rows.columns]
    #     latex_table += f"{qui_debt_at} & " + " & ".join(coeffs) + " \\\\\n"

    # latex_table += "\\midrule\n"
    # latex_table += "(t-statistics) & " + " & ".join(t_stats_with_stars) + " \\\\\n"
    # latex_table += "\\bottomrule\n"
    # latex_table += "\\end{tabular}"

    # print(latex_table)



def create_dlev_lev_port(df, quant_dlev, quant_intan, quant_lev, intan_subsample):
    quant_dlev_col = f'dlev_{quant_dlev}'
    quant_intan_col = f'intan_at_{quant_intan}'
    quant_lev_col = f'lev_{quant_lev}'

    df_subsample = df[df[quant_intan_col] == intan_subsample]
    # df_subsample = df

    # df.shape
    # df_subsample.shape
    grouped = df_subsample.groupby(['year_month', quant_lev_col, quant_dlev_col])
    # grouped.head(50)

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
    
    # Computing the Average Return for Each Group
    monthly_avg_returns = filtered_grouped['RET'].mean().reset_index()

    monthly_avg_returns.head(50)
    # Compute the Average Across All Months
    overall_avg_returns = monthly_avg_returns.groupby([quant_lev_col, quant_dlev_col])['RET'].mean().reset_index()
    # avr_returns_by_qui_debt_at.head(50)
    # pivot_df = overall_avg_returns.pivot(index='qui_d_debt_at', columns='qua_intan', values='ret_3mo_lead1')
    pivot_df = overall_avg_returns.pivot(index = quant_dlev_col, columns = quant_lev_col, values='RET')

    # Calculate the difference between the first and fifth rows
    difference_row = pivot_df.iloc[0] - pivot_df.iloc[quant_dlev - 1]
    difference_row.name = '1-{quant_dlev} Difference'

    # Step 7: Append the difference row to the pivoted DataFrame
    pivot_df = pivot_df._append(difference_row)

    print(pivot_df)

    ######################
    # Calculating t-stats
    ######################

    pivot_t_stat = monthly_avg_returns.pivot_table(values='RET', index=['year_month', quant_lev_col], columns = quant_dlev_col).reset_index()
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
    
    
    # # Extract the last row (t-stats) and the rows with coefficients
    # t_stats_row = pivot_df_with_t_stats.iloc[-1]
    # avr_rows = pivot_df_with_t_stats.iloc[:-1]

    # # Add stars to t-stats
    # t_stats_with_stars = [f"({t_stat:.2f}){add_stars(t_stat)}" for t_stat in t_stats_row]

    # # Create LaTeX table string
    # latex_table = "\\begin{tabular}{cccccc}\n"
    # latex_table += "\\toprule\n"
    # latex_table += "qui\\_debt\\_at & 1 & 2 & 3 & 4 & 5\\\\\n"
    # latex_table += "\\midrule\n"

    # for index, row in avr_rows.iterrows():
    #     ter_debt_at = row.name
    #     coeffs = [f"{row[col]:.2f}" for col in avr_rows.columns]
    #     latex_table += f"{ter_debt_at} & " + " & ".join(coeffs) + " \\\\\n"

    # latex_table += "\\midrule\n"
    # latex_table += "(t-statistics) & " + " & ".join(t_stats_with_stars) + " \\\\\n"
    # latex_table += "\\bottomrule\n"
    # latex_table += "\\end{tabular}"

    # print(latex_table)
    
    
    # Function to perform regression on a constant
def regress_on_constant(df):
    df = df.dropna(subset=['ret_diff'])  # Remove rows with NaN values in 'RET_diff'
    X = sm.add_constant(pd.Series([1]*len(df), index=df.index))  # Add a constant term (array of ones) with the same index as df
    model = sm.OLS(df['ret_diff'], X).fit()
    return model