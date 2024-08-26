import pandas as pd
import numpy as np
from tabulate import tabulate
from linearmodels.panel.model import FamaMacBeth
from statsmodels.api import OLS
import statsmodels.api as sm
import sys
sys.path.append('code/lev_stock/portfolio/')
sys.path.append('code/lev_stock/invest_strat/')
import utils as ut
import portfolio as pf


# def table_ff_dlev(df, quant_dlev, quant_intan, subsample, filename):
#     df_copy = df.copy()
#     if subsample == 'hint':
#         df_subsample = df_copy[df_copy['quant_intan'] == quant_intan]
#     elif subsample == 'lint':
#         df_subsample = df_copy[df_copy['quant_intan'] == 1]
#     else:
#         df_subsample = df_copy

#     df_subsample['ret_lead1_rf'] = df_subsample['ret_lead1'] - (df_subsample['rf']/100)
#     df_subsample['ret_lead2_rf'] = df_subsample['ret_lead2'] - (df_subsample['rf']/100)
#     quant_dlev_col = f'dlev_{quant_dlev}'
#     df_subsample = df_subsample.dropna(subset=[quant_dlev_col])
    
#     # Prepare return data for regressions
#     returns = df_subsample.groupby(['year_month', quant_dlev_col])[['ret_lead1_rf', 'ret_lead2_rf']].mean() * 100  # converting to percentage
#     returns = returns.reset_index()

#     factor_sets = {
#         'CAPM': ['mkt_rf'],
#         '3-factor': ['mkt_rf', 'smb', 'hml'],
#         '4-factor': ['mkt_rf', 'smb', 'hml', 'mom'],
#         '5-factor': ['mkt_rf', 'smb', 'hml', 'rmw', 'cma']
#     }
#     factors = df_subsample.groupby(['year_month', quant_dlev_col])[['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom']].first().reset_index()

#     regression_results = {}
#     for portfolio in ['ret_lead1_rf', 'ret_lead2_rf']:
#         # Pivot the returns
#         pivot_df = returns.pivot(index='year_month', columns=quant_dlev_col, values=portfolio)
        
#         # Calculate the difference
#         difference = pivot_df.iloc[:, 0] - pivot_df.iloc[:, -1]
#         difference.name = f'Difference 1 - {quant_dlev}'
        
#         # Add the difference to the pivot_df
#         pivot_df = pd.concat([pivot_df, difference], axis=1)
        
#         portfolio_results = {}
#         for column in pivot_df.columns:
#             quantile_results = {}
#             y = pivot_df[column]
            
#             for model_name, factor_set in factor_sets.items():
#                 if column == f'Difference 1 - {quant_dlev}':
#                     # For the difference column, use the factors from the last quantile
#                     X = factors[factors[quant_dlev_col] == pivot_df.columns[-2]][['year_month'] + factor_set].set_index('year_month')
#                 else:
#                     X = factors[factors[quant_dlev_col] == column][['year_month'] + factor_set].set_index('year_month')
                
#                 X = sm.add_constant(X)
#                 model = sm.OLS(y, X).fit()
#                 quantile_results[model_name] = model
            
#             portfolio_results[column] = quantile_results
        
#         regression_results[portfolio] = portfolio_results

#     #Generate LaTeX tables
#     for portfolio in returns.columns:
#         portfolio_results = {q: results[portfolio] for q, results in regression_results.items()}
#         latex_table = create_latex_table(portfolio_results, portfolio, filename)
#         print(f"LaTeX table for {portfolio}:")
#         print(latex_table)
#         print("\n")

#     # for portfolio, results in regression_results.items():
#     #     latex_table = create_latex_table(results, portfolio, filename, quant_dlev)
#     #     print(f"LaTeX table for {portfolio}:")
#     #     print(latex_table)
#     #     print("\n")
    
#     return regression_results

def table_ff_dlev(df, quant_dlev, quant_intan, subsample, filename):
    df_copy = df.copy()
    quant_intan_col = f'intan_at_{quant_intan}'
    
    if subsample == 'hint':
        df_subsample = df_copy[df_copy[quant_intan_col] == quant_intan]
    elif subsample == 'lint':
        df_subsample = df_copy[df_copy[quant_intan_col] == 1]
    else:
        df_subsample = df_copy
    
    
###########################################
# Applying factor models to each portfolio
###########################################

    df_subsample['ret_lead1_rf'] = df_subsample['ret_lead1'] - (df_subsample['rf']/100)
    df_subsample['ret_lead2_rf'] = df_subsample['ret_lead2'] - (df_subsample['rf']/100)
    quant_dlev_col = f'dlev_{quant_dlev}'
    df_subsample = df_subsample.dropna(subset=[quant_dlev_col])
    
    ret_lead1_rf_avg = df_subsample.groupby(['year_month', quant_dlev_col])['ret_lead1_rf'].mean() #float64
    ret_lead2_rf_avg = df_subsample.groupby(['year_month', quant_dlev_col])['ret_lead2_rf'].mean()
    returns = pd.concat([ret_lead1_rf_avg], axis=1) # appending the two columns together # dataframe
    # returns = pd.concat([ret_lead1_rf_avg, ret_lead2_rf_avg], axis=1) # appending the two columns together
    returns = returns*100 # converting to percentage
    
    pivot_df = ret_lead1_rf_avg.unstack(level=quant_dlev_col)
    difference_row = pivot_df[1] - pivot_df[quant_dlev]
    difference_row.name = quant_dlev+1 #f'1-{quant_dlev} Difference'
    pivot_df[difference_row.name] = difference_row
    
    melted_df = pivot_df.reset_index().melt(id_vars='year_month', var_name=quant_dlev_col, value_name='ret_lead1_rf')
    melted_df = melted_df.set_index(['year_month', 'dlev_5'])
    melted_df = melted_df*100
    
    factor_sets = [
        ['mkt_rf'], #CAPM
        ['mkt_rf', 'smb', 'hml'], # 3-factor
        ['mkt_rf', 'smb', 'hml', 'mom'], # 4-factor
        ['mkt_rf', 'smb', 'hml', 'rmw', 'cma'] # 5-factor
    ] #list
    
    factors = df_subsample.groupby(['year_month', quant_dlev_col])[['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom']].first()
    # Add the difference category to factors
    factors_pivot = factors.unstack(level=quant_dlev_col)
    category_5_factors = factors_pivot.xs(quant_dlev, axis=1, level=1)
    category_6_factors = category_5_factors.copy()
    category_6_factors.columns = pd.MultiIndex.from_product([category_6_factors.columns, [6]])
    factors_with_category_6 = pd.concat([factors_pivot, category_6_factors], axis=1)
    factors = factors_with_category_6.stack(level=1)
    factors.index = factors.index.set_names(quant_dlev_col, level=1)  # Add this line to name the categories column
    
    #factors = factors/100
     
    # factors = df_subsample[['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom']]
    df_subsample[quant_dlev_col] = df_subsample[quant_dlev_col].cat.add_categories([6])
    
    regression_results = {} #dictionary
    for dlev_value in df_subsample[quant_dlev_col].cat.categories: #categoricalDtype (1 through 5)+1 (the last is 1-5 difference, called 6).
        dlev_results = {}
        for portfolio in melted_df.columns: #numpy object
            portfolio_results = {}
            for i, factor_set in enumerate(factor_sets): #enumarate object. i goes from 0 to 3, factor_set is each list in factor_sets
                y = melted_df.loc[melted_df.index.get_level_values(quant_dlev_col) == dlev_value, portfolio] # takes the returns (portfolio) for the specific quantile (dlev_value)
                X = factors.loc[factors.index.get_level_values(quant_dlev_col) == dlev_value, factor_set] #index is an attribute of the dataframe. get_level_values is a method of the index object (not the dataframe)
                X = sm.add_constant(X)
                model = OLS(y, X).fit() #
                portfolio_results[f'model_{i+1}'] = model # this dictionary has 4 keys, each key is a model
            dlev_results[portfolio] = portfolio_results # this dictionary has 1 key, because we're only using ret_lead1_rf
        regression_results[dlev_value] = dlev_results # this dictionary has 5 keys, each key is a quantile. The total number of outcomes is 4x5 = 20 (each output matrix has 20 entries)
    
    # regression_results = {} #dictionary
    # for dlev_value in df_subsample[quant_dlev_col].unique(): #categoricalDtype (1 through 5)+1 (the last is 1-5 difference, called 6).
    #     dlev_results = {}
    #     for portfolio in returns.columns: #numpy object
    #         portfolio_results = {}
    #         for i, factor_set in enumerate(factor_sets): #enumarate object. i goes from 0 to 3, factor_set is each list in factor_sets
    #             y = returns.loc[returns.index.get_level_values(quant_dlev_col) == dlev_value, portfolio] # takes the returns (portfolio) for the specific quantile (dlev_value)
    #             X = factors.loc[factors.index.get_level_values(quant_dlev_col) == dlev_value, factor_set] #index is an attribute of the dataframe. get_level_values is a method of the index object (not the dataframe)
    #             X = sm.add_constant(X)
    #             model = OLS(y, X).fit() #
    #             portfolio_results[f'model_{i+1}'] = model # this dictionary has 4 keys, each key is a model
    #         dlev_results[portfolio] = portfolio_results # this dictionary has 1 key, because we're only using ret_lead1_rf
    #     regression_results[dlev_value] = dlev_results # this dictionary has 5 keys, each key is a quantile. The total number of outcomes is 4x5 = 20 (each output matrix has 20 entries)
    
    
    # for dlev_value, dlev_results in regression_results.items():
    #     print(f"Results for {quant_dlev_col} = {dlev_value}")
    #     for portfolio, portfolio_results in dlev_results.items():
    #         print(f"\nPortfolio: {portfolio}")
    #         for model_name, model in portfolio_results.items():
    #             print(f"\n{model_name} Summary:")
    #             print(model.summary().tables[1])  # Print coefficient table
    #     print("\n" + "="*50 + "\n")
        
    for portfolio in returns.columns:
        portfolio_results = {q: results[portfolio] for q, results in regression_results.items()} #dictionary comprehension, which creates a dictionary with 5 keys, each key is a quantile. .items() is a method of the dictionary object that creates key-value tuple pairs
        latex_table = create_latex_table(portfolio_results, portfolio, filename)
        print(f"LaTeX table for {portfolio}:")
        print(latex_table)
        print("\n")
        
    return regression_results
        
        
# def create_latex_table(portfolio_results, portfolio_name, filename):
#     models = ['CAPM', '3-factor', '4-factor', '5-factor']
#     quantiles = sorted([q for q in portfolio_results.keys() if q != 'Q5 - Q1 Difference'])
#     num_quantiles = len(quantiles)

#     latex_code = [
#         r"\begin{table}[htbp]",
#         r"\centering",
#         r"\begin{tabular}{l" + "c" * (num_quantiles + 1) + "}",
#         r"\hline",
#         "Model & " + " & ".join([f"Q{i+1}" for i in range(num_quantiles)]) + " & Q5 - Q1 \\\\",
#         r"\hline"
#     ]

#     for i, model in enumerate(models):
#         model_results = [
#             format_coefficient(
#                 portfolio_results[q][f'model_{i+1}'].params['const'],
#                 portfolio_results[q][f'model_{i+1}'].tvalues['const'],
#                 portfolio_results[q][f'model_{i+1}'].pvalues['const']
#             ) for q in quantiles
#         ]
        
#         # Add the difference column
#         diff_result = format_coefficient(
#             portfolio_results['Q5 - Q1 Difference'][f'model_{i+1}'].params['const'],
#             portfolio_results['Q5 - Q1 Difference'][f'model_{i+1}'].tvalues['const'],
#             portfolio_results['Q5 - Q1 Difference'][f'model_{i+1}'].pvalues['const']
#         )
#         model_results.append(diff_result)

#         latex_code.append(model + " & " + " & ".join([f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}} {result} \\end{{tabular}}" for result in model_results]) + r" \\[6pt]")

#     latex_code.extend([
#         r"\hline",
#         r"\end{tabular}",
#         f"\\caption{{Regression Results for {portfolio_name}}}",
#         r"\end{table}"
#     ])

#     latex_table = "\n".join(latex_code)
#     create_latex_table_file(latex_table, filename)

#     return latex_table

def create_latex_table(portfolio_results, portfolio_name, filename):
    models = ['CAPM', '3-factor', '4-factor', '5-factor']
    quantiles = sorted(portfolio_results.keys())
    
    latex_code = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        "Model & " + " & ".join([f"{i+1}" for i in range(len(quantiles))]) + r" \\",
        r"\hline"
    ]
    
    for i, model in enumerate(models):
        model_results = [format_coefficient(
            portfolio_results[q][f'model_{i+1}'].params['const'],
            portfolio_results[q][f'model_{i+1}'].tvalues['const'],
            portfolio_results[q][f'model_{i+1}'].pvalues['const']
        ) for q in quantiles]
        
        # latex_code.append(model + " & " + " & ".join(model_results) + r" \\")
        latex_code.append(model + " & " + " & ".join([f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}} {result} \\end{{tabular}}" for result in model_results]) + r" \\[6pt]")

        # if i < len(models) - 1:
        #     latex_code.append(r"\hline")
    
    latex_code.extend([
        r"\hline",
        r"\end{tabular}",
        f"\\caption{{Regression Results for {portfolio_name}}}",
        # r"\begin{tablenotes}",
        # r"\small",
        # r"\item Note: $^{*}p<0.1$; $^{**}p<0.05$; $^{***}p<0.01$",
        # r"\end{tablenotes}",
        r"\end{table}"
    ])
    
    # return "\n".join(latex_code)
    latex_table = "\n".join(latex_code)
    create_latex_table_file(latex_table, filename)
                                                            
def format_coefficient(coef, t_stat, p_value):
    stars = pf.get_significance_stars(p_value)
    return f"{coef:.2f} \\\\ ({t_stat:.2f}){stars}"


def get_significance_stars(p_value):
    if p_value < 0.01:
        return r'$^{***}$'
    elif p_value < 0.05:
        return r'$^{**}$'
    elif p_value < 0.1:
        return r'$^{*}$'
    else:
        return ''
    
    
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
