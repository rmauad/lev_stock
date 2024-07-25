import pandas as pd
import sys
sys.path.append('code/lev_stock/portfolio/')
import datacleaning as dc

intan = pd.read_csv('data/csv/int_xsec.csv') # from https://github.com/edwardtkim/intangiblevalue/tree/main/output
df = pd.read_csv("data/csv/comp_fundq.csv") # origianl Compustat
crsp = pd.read_csv('data/csv/crsp_full.csv') # original CRSP
ff = pd.read_csv('data/csv/F-F_Research_Data_Factors.CSV') # from Kenneth French's website
pt = pd.read_csv('data/csv/peterstaylor.csv')

comp_intan = dc.merge_comp_intan_epk(df, intan)
cc = dc.merge_crsp_comp(comp_intan, crsp, ff) # also merges with Fama-French factors
cc_pt = dc.merge_pt(cc, pt)
betas = dc.calc_beta(cc_pt)
df_fm = dc.prep_fm(cc_pt, betas)

save = df_fm.to_feather('data/feather/df_fm.feather')
