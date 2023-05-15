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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def descriptors(sub_prep_before, currency):

    last_value = sub_prep_before.loc[sub_prep_before["DATE"] == sub_prep_before["DATE"].max(), f"CLOSE_{currency}"].values[0]/sub_prep_before[f"CLOSE_{currency}"].mean()
    vol_before = sub_prep_before[f"CLOSE_{currency}"].std()/sub_prep_before[f"CLOSE_{currency}"].mean()
    mini_before = sub_prep_before[f"CLOSE_{currency}"].min()/sub_prep_before[f"CLOSE_{currency}"].mean()
    maxi_before = sub_prep_before[f"CLOSE_{currency}"].max()/sub_prep_before[f"CLOSE_{currency}"].mean()
    skew_before = sub_prep_before[f"CLOSE_{currency}"].skew()/sub_prep_before[f"CLOSE_{currency}"].mean()
    
    mini_before = (last_value - mini_before)/mini_before
    maxi_before = (last_value - maxi_before)/maxi_before
    # first_value = sub_prep_before.loc[sub_prep_before["DATE"] == sub_prep_before["DATE"].min(), f"CLOSE_{currency}"].values[0]
    # 
    # delta_before = (first_value - last_value) / first_value

    x = range(sub_prep_before.shape[0])
    y = sub_prep_before[f"CLOSE_{currency}"].tolist()
    
    # adding the constant term
    x = sm.add_constant(x)
    result = sm.OLS(y, x).fit()

    trend_before = -1*result.params[1]*30*100/sub_prep_before[f"CLOSE_{currency}"].mean()

    return [vol_before, mini_before, maxi_before, skew_before, trend_before]

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
        date_start = pd.to_datetime("2021-05-15", format = "%Y-%m-%d")

        for i in tqdm.tqdm(range(int(133))):
            date_end = date_start + timedelta(days=60)
            sub_prep_before = prepared.loc[prepared["DATE"].between(date_start, date_end)]
            moments_before = descriptors(sub_prep_before, currency)
            moments_during = descriptors(prepared.loc[prepared["DATE"].between(date_end, date_end + timedelta(days=60))], currency)
            
            strat = MainStrategy(configs=data_prep.configs, 
                                start_date=date_end,
                                end_date=date_end + timedelta(days=60))
            results = strat.strategy_1_lags_comparison(prepared, currency=currency)

            liste_results.append([currency, date_start, date_end] + moments_before + moments_during + results.iloc[-1].tolist()[1:])
            date_start = date_start + timedelta(days=5)
    
    return liste_results, data_prep, prepared

if __name__ == "__main__":
    liste_results, data_prep, prepared = main()

    df = pd.DataFrame(liste_results, columns=["CURRENCY", "DATE_START", "DATE_END", "STD_BF", "MIN_BF", "MAX_BF", "SKEW_BF", "DELTA_BF", "STD", "MIN", "MAX", "SKEW", "DELTA", "PNL_5", "PNL_7", "PNL_10", "PNL_15", "PNL_MEAN_LAGS", "PNL_MIX_MATCH"])
    df = df.loc[~df["DELTA"].isnull()]
    df = df.loc[~df["DELTA_BF"].isnull()]

    df["DELTA_BF"] = df["DELTA_BF"].clip(-3, 3)
    df["STD_BF"] = df["STD_BF"].clip(0, 0.5)
    df["MAX_BF"] = df["MAX_BF"].clip(1, 3)
    df["MIN_BF"] = df["MAX_BF"].clip(0.3, 1)
    df["SKEW_BF"] = df["SKEW_BF"].clip(-20, 20)

    df["DELTA"] = df["DELTA"].clip(-3, 3)
    df["STD"] = df["STD"].clip(0, 0.5)
    df["MAX"] = df["MAX"].clip(1, 3)
    df["MIN"] = df["MAX"].clip(0.3, 1)
    df["SKEW"] = df["SKEW"].clip(-20, 20)

    kmeans = KMeans(n_clusters=3)
    scaler=StandardScaler().fit(df[["STD_BF", "SKEW_BF", "DELTA_BF"]])
    X= scaler.transform(df[["STD_BF", "SKEW_BF", "DELTA_BF"]])

    kmeans.fit(X)
    k_means_labels = kmeans.labels_
    k_means_cluster_centers = kmeans.cluster_centers_
    df["CLUSTER_BF"] = k_means_labels

    XX = scaler.transform(df[["STD", "SKEW", "DELTA"]])
    df["CLUSTER_AF"] = kmeans.predict(XX)

    fig, ax = plt.subplots(figsize=(20,10)) 
    colors = ["#4EACC5", "#FF9C34", "#4E9A06", "pink", "red"]
    
    # KMeans
    # ax = fig.add_subplot(1,1,1)
    for k, col in zip(range(3), colors):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        ax.plot(X[my_members, 2], X[my_members, 0], "w", markerfacecolor=col, marker=".")
        ax.plot(
            cluster_center[2],
            cluster_center[0],
            "o",
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=6,
        )
    ax.set_title("KMeans")
    
    # df = df.loc[df["DATE_END"] <= "2023-05-15"]

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