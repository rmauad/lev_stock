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
#     quant_intan_col = f'intan_at_{quant_intan}'
    
#     if subsample == 'hint':
#         df_subsample = df_copy[df_copy[quant_intan_col] == quant_intan]
#     elif subsample == 'lint':
#         df_subsample = df_copy[df_copy[quant_intan_col] == 1]
#     else:
#         df_subsample = df_copy
    
#     df_subsample['ret_lead1_rf'] = df_subsample['ret_lead1'] - (df_subsample['rf']/100)
#     quant_dlev_col = f'dlev_{quant_dlev}'
#     df_subsample = df_subsample.dropna(subset=[quant_dlev_col])
    
#     # Equal-weighted returns
#     ret_lead1_rf_avg_ew = df_subsample.groupby(['year_month', quant_dlev_col])['ret_lead1_rf'].mean()
    
#     # Value-weighted returns
#     grouped = df_subsample.groupby(['year_month', quant_dlev_col])
#     ret_lead1_rf_avg_vw = grouped.apply(lambda x: (x['ret_lead1_rf'] * x['me']).sum() / x['me'].sum()).reset_index(name='ret_lead1_rf')    
#     ret_lead1_rf_avg_vw = ret_lead1_rf_avg_vw.set_index(['year_month', quant_dlev_col])['ret_lead1_rf']
    
#     returns_ew = pd.DataFrame(ret_lead1_rf_avg_ew*100, columns=['ret_lead1_rf'])
#     returns_vw = pd.DataFrame(ret_lead1_rf_avg_vw*100, columns=['ret_lead1_rf'])
    
#     # Calculate differences for both EW and VW
#     for returns in [returns_ew, returns_vw]:
#         pivot_df = returns['ret_lead1_rf'].unstack(level=quant_dlev_col)
#         difference_row = pivot_df[1] - pivot_df[quant_dlev]
#         difference_row.name = quant_dlev+1
#         pivot_df[difference_row.name] = difference_row
#         returns = pivot_df.stack().reset_index()
#         returns.columns = ['year_month', quant_dlev_col, 'ret_lead1_rf']
#         returns = returns.set_index(['year_month', quant_dlev_col])
    
#     factor_sets = [
#         ['mkt_rf'], #CAPM
#         ['mkt_rf', 'smb', 'hml'], # 3-factor
#         ['mkt_rf', 'smb', 'hml', 'mom'], # 4-factor
#         ['mkt_rf', 'smb', 'hml', 'rmw', 'cma'] # 5-factor
#     ]
    
#     factors = df_subsample.groupby(['year_month', quant_dlev_col])[['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom']].first()
#     # Add the difference category to factors
#     factors_pivot = factors.unstack(level=quant_dlev_col)
#     category_5_factors = factors_pivot.xs(quant_dlev, axis=1, level=1)
#     category_6_factors = category_5_factors.copy()
#     category_6_factors.columns = pd.MultiIndex.from_product([category_6_factors.columns, [6]])
#     factors_with_category_6 = pd.concat([factors_pivot, category_6_factors], axis=1)
#     factors = factors_with_category_6.stack(level=1)
#     factors.index = factors.index.set_names(quant_dlev_col, level=1)  # Add this line to name the categories column


#     # factors = df_subsample.groupby(['year_month', quant_dlev_col])[['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom']].first()
#     factors = factors.reset_index().set_index(['year_month', quant_dlev_col])
    
#     regression_results_ew = {}
#     regression_results_vw = {}
    
#     for dlev_value in df_subsample[quant_dlev_col].cat.categories:
#         dlev_results_ew = {}
#         dlev_results_vw = {}
#         for portfolio in returns_ew.columns:
#             portfolio_results_ew = {}
#             portfolio_results_vw = {}
#             for i, factor_set in enumerate(factor_sets):
#                 y_ew = returns_ew.loc[returns_ew.index.get_level_values(quant_dlev_col) == dlev_value, portfolio]
#                 y_vw = returns_vw.loc[returns_vw.index.get_level_values(quant_dlev_col) == dlev_value, portfolio]
#                 X = factors.loc[factors.index.get_level_values(quant_dlev_col) == dlev_value, factor_set]
#                 X = sm.add_constant(X)
#                 model_ew = OLS(y_ew, X).fit()
#                 model_vw = OLS(y_vw, X).fit()
#                 portfolio_results_ew[f'model_{i+1}'] = model_ew
#                 portfolio_results_vw[f'model_{i+1}'] = model_vw
#             dlev_results_ew[portfolio] = portfolio_results_ew
#             dlev_results_vw[portfolio] = portfolio_results_vw
#         regression_results_ew[dlev_value] = dlev_results_ew
#         regression_results_vw[dlev_value] = dlev_results_vw
    
#     for portfolio in returns_ew.columns:
#         portfolio_results_ew = {q: results[portfolio] for q, results in regression_results_ew.items()}
#         portfolio_results_vw = {q: results[portfolio] for q, results in regression_results_vw.items()}
#         latex_table = create_latex_table(portfolio_results_ew, portfolio_results_vw, portfolio, filename)
#         print(f"LaTeX table for {portfolio}:")
#         print(latex_table)
#         print("\n")
    
#     return regression_results_ew, regression_results_vw

# def create_latex_table(portfolio_results_ew, portfolio_results_vw, portfolio_name, filename):
#     models = ['CAPM', '3-factor', '4-factor', '5-factor']
#     quantiles = sorted(portfolio_results_ew.keys())
    
#     latex_code = [
#         r"\begin{table}[htbp]",
#         r"\centering",
#         r"\begin{tabular}{lcccccc}",
#         r"\hline",
#         r"\multicolumn{7}{c}{Panel A: Equal-Weighted} \\",
#         r"\hline",
#         "Model & " + " & ".join([f"{i+1}" for i in range(len(quantiles))]) + r" \\",
#         r"\hline"
#     ]
    
#     for i, model in enumerate(models):
#         model_results = [format_coefficient(
#             portfolio_results_ew[q][f'model_{i+1}'].params['const'],
#             portfolio_results_ew[q][f'model_{i+1}'].tvalues['const'],
#             portfolio_results_ew[q][f'model_{i+1}'].bse['const'],
#             portfolio_results_ew[q][f'model_{i+1}'].conf_int().loc['const'].tolist(),
#             portfolio_results_ew[q][f'model_{i+1}'].pvalues['const']
#         ) for q in quantiles]
        
#         latex_code.append(model + " & " + " & ".join([f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}} {result} \\end{{tabular}}" for result in model_results]) + r" \\[6pt]")

#     latex_code.extend([
#         r"\hline",
#         r"\multicolumn{7}{c}{Panel B: Value-Weighted} \\",
#         r"\hline",
#         "Model & " + " & ".join([f"{i+1}" for i in range(len(quantiles))]) + r" \\",
#         r"\hline"
#     ])

#     for i, model in enumerate(models):
#         model_results = [format_coefficient(
#             portfolio_results_vw[q][f'model_{i+1}'].params['const'],
#             portfolio_results_vw[q][f'model_{i+1}'].tvalues['const'],
#             portfolio_results_vw[q][f'model_{i+1}'].bse['const'],
#             portfolio_results_vw[q][f'model_{i+1}'].conf_int().loc['const'].tolist(),
#             portfolio_results_vw[q][f'model_{i+1}'].pvalues['const']
#         ) for q in quantiles]
        
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
    
    ret_lead1_rf_avg_ew = df_subsample.groupby(['year_month', quant_dlev_col])['ret_lead1_rf'].mean() #float64
    
    grouped = df_subsample.groupby(['year_month', quant_dlev_col])
    ret_lead1_rf_avg_vw = grouped.apply(lambda x: (x['ret_lead1_rf'] * x['me']).sum() / x['me'].sum()).reset_index(name=f'ret_lead1_rf')    
    ret_lead1_rf_avg_vw = ret_lead1_rf_avg_vw.set_index(['year_month', quant_dlev_col])['ret_lead1_rf']
    # ret_lead2_rf_avg = df_subsample.groupby(['year_month', quant_dlev_col])['ret_lead2_rf'].mean()
    
    #########################
    # Equal-weighted returns
    #########################
    returns_ew = pd.concat([ret_lead1_rf_avg_ew], axis=1) # appending the two columns together # dataframe
    # returns = pd.concat([ret_lead1_rf_avg, ret_lead2_rf_avg], axis=1) # appending the two columns together
    returns_ew = returns_ew*100 # converting to percentage
    
    #########################
    # Value-weighted returns
    #########################
    returns_vw = pd.concat([ret_lead1_rf_avg_vw], axis=1) # appending the two columns together # dataframe
    # returns = pd.concat([ret_lead1_rf_avg, ret_lead2_rf_avg], axis=1) # appending the two columns together
    returns_vw = returns_vw*100 # converting to percentage
    
    
    pivot_df_ew = ret_lead1_rf_avg_ew.unstack(level=quant_dlev_col)
    difference_row_ew = pivot_df_ew[1] - pivot_df_ew[quant_dlev]
    difference_row_ew.name = quant_dlev+1 #f'1-{quant_dlev} Difference'
    pivot_df_ew[difference_row_ew.name] = difference_row_ew
    
    melted_df_ew = pivot_df_ew.reset_index().melt(id_vars='year_month', var_name=quant_dlev_col, value_name='ret_lead1_rf')
    melted_df_ew = melted_df_ew.set_index(['year_month', 'dlev_5'])
    melted_df_ew = melted_df_ew*100
    
    pivot_df_vw = ret_lead1_rf_avg_vw.unstack(level=quant_dlev_col)
    difference_row_vw = pivot_df_vw[1] - pivot_df_vw[quant_dlev]
    difference_row_vw.name = quant_dlev+1 #f'1-{quant_dlev} Difference'
    pivot_df_vw[difference_row_vw.name] = difference_row_vw
    
    melted_df_vw = pivot_df_vw.reset_index().melt(id_vars='year_month', var_name=quant_dlev_col, value_name='ret_lead1_rf')
    melted_df_vw = melted_df_vw.set_index(['year_month', 'dlev_5'])
    melted_df_vw = melted_df_vw*100
    
    factor_sets = [
        ['mkt_rf'], #CAPM
        ['mkt_rf', 'smb', 'hml'], # 3-factor
        # ['mkt_rf', 'smb', 'hml', 'mom'], # 4-factor
        ['mkt_rf', 'smb', 'hml', 'rmw', 'cma'], # 5-factor
        ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom'] # 4-factor

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

     
    # factors = df_subsample[['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'mom']]
    df_subsample[quant_dlev_col] = df_subsample[quant_dlev_col].astype('category')
    df_subsample[quant_dlev_col] = df_subsample[quant_dlev_col].cat.add_categories([6])
    
    regression_results_ew = {} #dictionary
    regression_results_vw = {} #dictionary
    for dlev_value in df_subsample[quant_dlev_col].cat.categories: #categoricalDtype (1 through 5)+1 (the last is 1-5 difference, called 6).
        dlev_results_ew = {}
        dlev_results_vw = {}
        for portfolio in melted_df_ew.columns: #numpy object
            portfolio_results_ew = {}
            portfolio_results_vw = {}
            for i, factor_set in enumerate(factor_sets): #enumarate object. i goes from 0 to 3, factor_set is each list in factor_sets
                y_ew = melted_df_ew.loc[melted_df_ew.index.get_level_values(quant_dlev_col) == dlev_value, portfolio] # takes the returns (portfolio) for the specific quantile (dlev_value)
                y_vw = melted_df_vw.loc[melted_df_vw.index.get_level_values(quant_dlev_col) == dlev_value, portfolio] # takes the returns (portfolio) for the specific quantile (dlev_value)                
                X = factors.loc[factors.index.get_level_values(quant_dlev_col) == dlev_value, factor_set] #index is an attribute of the dataframe. get_level_values is a method of the index object (not the dataframe)
                X = sm.add_constant(X)
                model_ew = OLS(y_ew, X).fit() #
                model_vw = OLS(y_vw, X).fit() #
                portfolio_results_ew[f'model_{i+1}'] = model_ew # this dictionary has 4 keys, each key is a model
                portfolio_results_vw[f'model_{i+1}'] = model_vw # this dictionary has 4 keys, each key is a model
            dlev_results_ew[portfolio] = portfolio_results_ew # this dictionary has 1 key, because we're only using ret_lead1_rf
            dlev_results_vw[portfolio] = portfolio_results_vw # this dictionary has 1 key, because we're only using ret_lead1_rf
        regression_results_ew[dlev_value] = dlev_results_ew # this dictionary has 5 keys, each key is a quantile. The total number of outcomes is 4x5 = 20 (each output matrix has 20 entries)
        regression_results_vw[dlev_value] = dlev_results_vw # this dictionary has 5 keys, each key is a quantile. The total number of outcomes is 4x5 = 20 (each output matrix has 20 entries)
    
        
    for portfolio in returns_ew.columns:
        portfolio_results_ew = {q: results[portfolio] for q, results in regression_results_ew.items()} #dictionary comprehension, which creates a dictionary with 5 keys, each key is a quantile. .items() is a method of the dictionary object that creates key-value tuple pairs
        portfolio_results_vw = {q: results[portfolio] for q, results in regression_results_vw.items()} #dictionary comprehension, which creates a dictionary with 5 keys, each key is a quantile. .items() is a method of the dictionary object that creates key-value tuple pairs        
        latex_table = create_latex_table(portfolio_results_ew, portfolio_results_vw, portfolio, filename)
        print(f"LaTeX table for {portfolio}:")
        print(latex_table)
        print("\n")
        
    return regression_results_ew

def create_latex_table(portfolio_results_ew, portfolio_results_vw, portfolio_name, filename):
    models = ['CAPM', '3-factor', '5-factor', '6-factor']
    quantiles = sorted(portfolio_results_ew.keys())

    def format_coefficient(coef, se, conf_int, pval):
        stars = ''
        if pval <= 0.01:
            stars = r'$^{***}$'
        elif pval <= 0.05:
            stars = r'$^{**}$'
        elif pval <= 0.1:
            stars = r'$^{*}$'
        coef_str = f"{coef:.2f}{stars}"
        se_str = f"[{se:.2f}]"
        ci_str = f"({conf_int[0]:.2f}, {conf_int[1]:.2f})"
        return coef_str, se_str, ci_str

    latex_code = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l@{\hspace{8pt}}c@{\hspace{8pt}}c@{\hspace{8pt}}c@{\hspace{8pt}}c@{\hspace{8pt}}c@{\hspace{8pt}}c}",
        r"\toprule",
        r"& \multicolumn{5}{c}{Leverage change portfolio} \\",
        r"\cmidrule{2-7}",
        r" & 1 & 2 & 3 & 4 & 5 & Difference 1-5 \\",
        r"\hline",
        r" \multicolumn{7}{l}{\textit{Panel A: Equal-Weighted Portfolio}} \\",
        r"\hline \\"
    ]

    # Equal-Weighted Portfolio
    for i, model in enumerate(models):
        model_results = [format_coefficient(
            portfolio_results_ew[q][f'model_{i+1}'].params['const'],
            portfolio_results_ew[q][f'model_{i+1}'].bse['const'],
            portfolio_results_ew[q][f'model_{i+1}'].conf_int().loc['const'].tolist(),
            portfolio_results_ew[q][f'model_{i+1}'].pvalues['const']
        ) for q in quantiles]
        
        # Extract formatted coefficient, standard error, and confidence interval separately
        coef_row = [result[0] for result in model_results]
        se_row = [result[1] for result in model_results]
        ci_row = [result[2] for result in model_results]

        latex_code.append(f"{model} & " + " & ".join(coef_row) + r" \\")
        latex_code.append(" & " + " & ".join(se_row) + r" \\")
        latex_code.append(" & " + " & ".join(ci_row) + r" \\[3pt]")
        latex_code.append(r" & & & & & & \\[3pt]")  # Add spacing row

    # Divider and Value-Weighted Portfolio header
    latex_code.extend([
        r"\hline",
        r"\\[-8pt]",
        r"\multicolumn{7}{l}{\textit{Panel B: Value-Weighted Portfolio}} \\",
        r"\hline \\"
    ])

    # Value-Weighted Portfolio
    for i, model in enumerate(models):
        model_results = [format_coefficient(
            portfolio_results_vw[q][f'model_{i+1}'].params['const'],
            portfolio_results_vw[q][f'model_{i+1}'].bse['const'],
            portfolio_results_vw[q][f'model_{i+1}'].conf_int().loc['const'].tolist(),
            portfolio_results_vw[q][f'model_{i+1}'].pvalues['const']
        ) for q in quantiles]
        
        # Extract formatted coefficient, standard error, and confidence interval separately
        coef_row = [result[0] for result in model_results]
        se_row = [result[1] for result in model_results]
        ci_row = [result[2] for result in model_results]

        latex_code.append(f"{model} & " + " & ".join(coef_row) + r" \\")
        latex_code.append(" & " + " & ".join(se_row) + r" \\")
        latex_code.append(" & " + " & ".join(ci_row) + r" \\[3pt]")
        latex_code.append(r" & & & & & & \\[3pt]")  # Add spacing row

    latex_code.extend([
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{Regression Results for {portfolio_name}}}",
        r"\label{tab:regression_results}",
        r"\end{table}"
    ])

    latex_table = "\n".join(latex_code)
    create_latex_table_file(latex_table, filename)

    
# def create_latex_table(portfolio_results_ew, portfolio_results_vw, portfolio_name, filename):
#     models = ['CAPM', '3-factor', '4-factor', '5-factor']
#     quantiles = sorted(portfolio_results_ew.keys())
    
#     latex_code = [
#         r"\begin{table}[htbp]",
#         r"\centering",
#         r"\begin{tabular}{lcccccc}",
#         r"\multicolumn{7}{c}{\textbf{Panel A: Equal-Weighted Portfolio}} \\",
#         r"\hline",
#         "Model & " + " & ".join([f"{i+1}" for i in range(len(quantiles))]) + r" \\",
#         r"\hline"
#     ]
    
#     # Equal-Weighted Portfolio
#     for i, model in enumerate(models):
#         model_results_ew = [format_coefficient(
#             portfolio_results_ew[q][f'model_{i+1}'].params['const'],
#             portfolio_results_ew[q][f'model_{i+1}'].tvalues['const'],
#             portfolio_results_ew[q][f'model_{i+1}'].bse['const'],
#             portfolio_results_ew[q][f'model_{i+1}'].conf_int().loc['const'].tolist(),
#             portfolio_results_ew[q][f'model_{i+1}'].pvalues['const']
#         ) for q in quantiles]
#         latex_code.append(model + " & " + " & ".join([f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}} {result} \\end{{tabular}}" for result in model_results_ew]) + r" \\[6pt]")
    
#     latex_code.extend([
#         r"\hline",
#         r"\\",  # Add some vertical space between panels
#         r"\multicolumn{7}{c}{\textbf{Panel B: Value-Weighted Portfolio}} \\",
#         r"\hline",
#         "Model & " + " & ".join([f"{i+1}" for i in range(len(quantiles))]) + r" \\",
#         r"\hline"
#     ])
    
#     # Value-Weighted Portfolio
#     for i, model in enumerate(models):
#         model_results_vw = [format_coefficient(
#             portfolio_results_vw[q][f'model_{i+1}'].params['const'],
#             portfolio_results_vw[q][f'model_{i+1}'].tvalues['const'],
#             portfolio_results_vw[q][f'model_{i+1}'].bse['const'],
#             portfolio_results_vw[q][f'model_{i+1}'].conf_int().loc['const'].tolist(),
#             portfolio_results_vw[q][f'model_{i+1}'].pvalues['const']
#         ) for q in quantiles]
#         latex_code.append(model + " & " + " & ".join([f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}} {result} \\end{{tabular}}" for result in model_results_vw]) + r" \\[6pt]")
    
#     latex_code.extend([
#         r"\hline",
#         r"\end{tabular}",
#         f"\\caption{{Regression Results for {portfolio_name}}}",
#         r"\label{tab:regression_results}",  # Added label for easy referencing
#         r"\end{table}"
#     ])
    
#     latex_table = "\n".join(latex_code)
#     create_latex_table_file(latex_table, filename)
    

def format_coefficient(coef, t_stat, std_err, conf_int, p_value):
    stars = pf.get_significance_stars(p_value)
    # return f"{coef:.2f} \\\\ ({t_stat:.2f}){stars} \\\\ [{std_err:.2f}] \\\\ ({conf_int[0]:.2f}, {conf_int[1]:.2f})"
    return f"{coef:.2f}{stars} \\\\ [{std_err:.2f}] \\\\ ({conf_int[0]:.2f}, {conf_int[1]:.2f})"



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
