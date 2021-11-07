import seaborn as sns 
import numpy as np
import pandas as pd 
sns.set_style(style="dark")

from data_analysis.plot_analysis import var_vs_target
from utils.general_functions import weight_history


def analyse_variables(full):

    # trading done 6-20 UST
    var_vs_target(data=full, Y_label="TRADECOUNT_BTC", variable="HOUR", bins=24)

    # price lower between 2-6, higher at 22-23 ! -> vendre 22H, acheter 2H UST
    var_vs_target(data=full, Y_label="OPEN_BTC", variable="HOUR", bins=24)

    # price higher week-end + monday, lower end of week (thursday-friday)
    var_vs_target(data=full, Y_label="OPEN_BTC", variable="WEEK_DAY", bins=24)

    # end of year look lower than start of year -> not enough history -> don't use Month
    var_vs_target(data=full, Y_label="OPEN_BTC", variable="MONTH", bins=24)


def correlation_analysis(full):

    all_op = [x for x in full.columns if "OPEN" in x ]
    sns.heatmap(full[all_op].corr())


def delta_serie(serie : pd.Series, d1=1, d2=2):
    return serie.diff(d1) - serie.diff(d2) / serie.diff(d2) 


def prepare_currency_daily(full, currency="BTC", rolling=7):

    btc = full[["DATE", "WEEK_DAY", "HOUR", f"OPEN_{currency}", f"TRADECOUNT_{currency}"]]
    btc = btc.loc[~btc[f"OPEN_{currency}"].isnull()]
    btc.rename(columns={f"TRADECOUNT_{currency}" : "TRADECOUNT"}, inplace=True)
    btc["DT_HR"] = pd.to_datetime(btc['DATE'].astype(str) + ' ' +  btc['HOUR'].apply(lambda x: str(x).zfill(2) + ":00:00"))

    # create aggegated version
    agg = btc[["DATE", "TRADECOUNT"]].groupby("DATE").sum().reset_index()
    agg_currency = btc.loc[btc["HOUR"] == 0][["DATE", f"OPEN_{currency}", "WEEK_DAY"]]
    agg = pd.merge(agg, agg_currency, on="DATE", validate="1:1")
    agg = agg.sort_values("DATE", ascending=0)

    # create target 
    agg["TARGET_NORMALIZED"]= agg[f"OPEN_{currency}"].diff(1)*-100/ agg[f"OPEN_{currency}"]
    agg["TARGET"]= agg[f"OPEN_{currency}"].diff(1)*-1
    agg["BINARY_TARGET"]= (agg["TARGET"] > 0)*1

    # variation % last X days
    for i in range(1, 4*rolling +1):
        agg[f"D-{i}"] =agg[f"OPEN_{currency}"].diff(-i)*100/agg[f"OPEN_{currency}"].shift(-i)
    
    agg["MEAN_7"] = agg[[f"D-{x}" for x in range(1, rolling +1)]].mean(axis=1)
    agg["MEAN_14"] = agg[[f"D-{x}" for x in range(1, 2*rolling +1)]].mean(axis=1)
    agg["MEAN_28"] = agg[[f"D-{x}" for x in range(1, 4*rolling +1)]].mean(axis=1)

    agg["WEEK_DAY"] = agg["WEEK_DAY"].astype("category")

    # Tradecount 
    agg["TRADECOUNT"] = agg["TRADECOUNT"]/agg[f"OPEN_{currency}"]
    agg["TRADECOUNT"] = np.where(agg["TRADECOUNT"]==0, np.nan, agg["TRADECOUNT"])

    # history weight
    agg["WEIGHT"] = weight_history(agg, date_name="DATE", k=3)

    # rolling std 
    agg = agg.sort_values("DATE", ascending=1)
    agg[f"MEDIAN-{rolling}"] = agg[f"OPEN_{currency}"].rolling(rolling, min_periods=1).median() / agg[f"OPEN_{currency}"]
    agg[f"STD-{rolling}"] = agg[f"OPEN_{currency}"].rolling(rolling, min_periods=1).std() / agg[f"OPEN_{currency}"]
    agg[f"MIN-{rolling}"] = agg[f"OPEN_{currency}"].rolling(rolling, min_periods=1).min() / agg[f"OPEN_{currency}"]
    agg[f"MAX-{rolling}"] = agg[f"OPEN_{currency}"].rolling(rolling, min_periods=1).max() / agg[f"OPEN_{currency}"]
    agg[f"NBR_INCREMENTALS-{rolling}"] = agg[f"OPEN_{currency}"].diff(1).rolling(rolling, min_periods=1).apply(lambda x: sum(x > 0))
    agg[f"AVG_INCREMENTALS-{rolling}"] = agg[[f"D-{x}" for x in range(1, rolling +1)]].abs().mean(axis=1)

    agg[f"MEDIAN-{2*rolling}"] = agg[f"OPEN_{currency}"].rolling(2*rolling, min_periods=1).median() / agg[f"OPEN_{currency}"]
    agg[f"STD-{2*rolling}"] = agg[f"OPEN_{currency}"].rolling(2*rolling, min_periods=1).std() / agg[f"OPEN_{currency}"]
    agg[f"MIN-{2*rolling}"] = agg[f"OPEN_{currency}"].rolling(2*rolling, min_periods=1).min() / agg[f"OPEN_{currency}"]
    agg[f"MAX-{2*rolling}"] = agg[f"OPEN_{currency}"].rolling(2*rolling, min_periods=1).max() / agg[f"OPEN_{currency}"]
    agg[f"NBR_INCREMENTALS-{2*rolling}"] = agg[f"OPEN_{currency}"].diff(1).rolling(2*rolling, min_periods=1).apply(lambda x: sum(x > 0))
    agg[f"AVG_INCREMENTALS-{2*rolling}"] = agg[[f"D-{x}" for x in range(1, 2*rolling +1)]].abs().mean(axis=1)


    agg = agg.sort_values("DATE", ascending=0)

    # Q: Est ce que si ca augmente de plus de X% au cours de Y jours, baisse le lendemain ? 
    # inversement pour la baisse (trouver X et Y)

    return agg
