import pandas as pd 
import statsmodels.api as sm
import numpy as np

df = final_inputs["results"]
df = pd.read_csv(r"C:\Users\de larrard alexandre\Documents\repos_github\PEA\data\results\financials.csv", sep=";")
df.index = df["Unnamed: 0"]
del df["Unnamed: 0"]

y = df["_P/E_TTM"].astype(float)
y = np.where(y > 1000, 1000, y)

X = df[['%_TTM_NET_INCOME_BEFORE_EXTRA_ITEMS', '%_TTM_NET_PROFIT_MARGIN',
       '%_TTM_OPERATING_INCOME', '%_TTM_TAXES_&_EXTRA_IN_TOTAL_REVENUE',
       '%_TTM_TOTAL_REVENUE', '%_YOY_NET_INCOME_BEFORE_EXTRA_ITEMS',
       '%_YOY_OPERATING_INCOME',  'BALANCE_%_MARKET_CAP / TOTAL_ASSETS',
       'BALANCE_%_TOTAL_ASSET / TOTAL_LIABILITIES', 'CASH_ACQUISITION_OVER_ACTIVITY', 
       'CASH_FINANCE_OVER_ACTIVITY_CASH','LONG_TREND_NET_INCOME_BEFORE_EXTRA_ITEMS', 'LONG_TREND_OP_INCOME',
       'LONG_TREND_SHARE_SALES_GENERAL_IN_REVENUE', 'LONG_TREND_TOTAL_REVENUE', 'STOCK_% TREND_NET_INCOME_WO_EXTRA - STOCK_TREND', 
       'STOCK_%_52_WEEKS', 'TEAM_CEO_APPOINTED', '_P/E_Y-1', '_NET_INCOME_BEFORE_EXTRA_ITEMS']]

for col in X.columns:
    X[col] = np.where(X[col].isin([np.inf, -np.inf]), np.nan, X[col])
    X[col] = X[col].astype(float)
    X[col] = X[col].fillna(X[col].median())

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())