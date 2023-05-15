import pandas as pd
import time
import logging 
import tqdm
import os 
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
os.chdir("../")

from data_prep.data_preparation_crypto import PrepareCrytpo 
from strategy.strategie_1 import MainStrategy 
import statsmodels.api as sm

def main():

    # load data
    data_prep = PrepareCrytpo()

    logging.info("Starting data / strategy execution")
    datas = data_prep.load_share_price_data()

    # data preparation 
    prepared = data_prep.aggregate_crypto_price(datas)
    
    liste_results= []
    for currency in data_prep.currencies:

        # strategy deduce buy / sell per currency
        date_start = pd.to_datetime("2021-08-01", format = "%Y-%m-%d")
        print(currency)

        for i in tqdm.tqdm(range(int(130))):
            date_end = date_start + timedelta(days=60)
            # sub_prep_before = prepared.loc[prepared["DATE"].between(date_start -timedelta(days=90), date_start)]
            # vol = sub_prep_before[f"CLOSE_{currency}"].std()/sub_prep_before[f"CLOSE_{currency}"].mean()
            
            # x = range(sub_prep_before.shape[0])
            # y = sub_prep_before[f"CLOSE_{currency}"].tolist()
            
            # # adding the constant term
            # x = sm.add_constant(x)
            # result = sm.OLS(y, x).fit()

            # trend = -1*result.params[1]*30*100/sub_prep_before[f"CLOSE_{currency}"].mean()

            strat = MainStrategy(configs=data_prep.configs, 
                                start_date=date_start,
                                end_date=date_end)
            results = strat.strategy_1_lags_comparison(prepared, currency=currency)

            liste_results.append([currency, date_start, date_end] + results.iloc[-1].tolist()[1:])
            date_start = date_start + timedelta(days=4)
    
    return liste_results, data_prep, prepared

if __name__ == "__main__":

    liste_results, data_prep, prepared = main()

    df = pd.DataFrame(liste_results, columns=["CURRENCY", "DATE_START", "DATE_END", "PNL_5", "PNL_7", "PNL_10", "PNL_15", "PNL_MEAN_LAGS", "PNL_MIX_MATCH"])
    df = df.loc[df["DATE_END"] <= "2023-05-15"]

#     for currency in data_prep.currencies:
#         btc = df.loc[df["CURRENCY"] == currency]
#         print(btc.mean())

#     prep = prepared
    
#     for cur in data_prep.currencies:
#         prep = prepared.copy()
#         prep["DATE"] = prep["DATE"].dt.round("D")
#         prep = prep[["DATE", f"CLOSE_{cur}"]].groupby("DATE").mean().reset_index()
#         fig, ax = plt.subplots(figsize=(20,10)) 
#         df.loc[df["DATE_START"]>="2021-12-01"].loc[df["CURRENCY"] == cur].set_index("DATE_END")[["PNL_7", "PNL_15", "PNL_30", "PNL_45", "PNL_MEAN_LAGS", "PNL_MIX_MATCH"]].plot(title=cur, ax=ax)
#         prep.loc[prep["DATE"].between("2022-02-01",df["DATE_END"].max())].set_index("DATE").plot(ax=ax, secondary_y=True, color='r')
#         plt.show()