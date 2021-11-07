import pandas as pd 
from utils.general_functions import smart_column_parser


def clean_hour(x):

    x = str(x) 
    if "AM" in x:
        x = int(x.split("-")[0])
    elif "PM" in x:
        x = int(x.split("-")[0]) + 12 
    elif ":" in x:
        x = int(x.split(":")[0])
    else:
        print(x)

    if int(x) == 24:
        return 0 
    else:
        return x


def clean_share_price(datas):

    full = pd.DataFrame()

    for k in datas.keys():
        datas[k].columns = smart_column_parser(datas[k].columns)

        df = datas[k].copy()
        df["HOUR"] = df["DATE"].apply(lambda x: clean_hour(str(x).split()[1]))
        df["DATE"] = pd.to_datetime(df["DATE"].apply(lambda x: str(x).split()[0]), format="%Y-%m-%d")
        df = df.drop_duplicates(["DATE", "HOUR"])
        df["RANGE"] = df["HIGH"] - df["LOW"]
        df.rename(columns={"OPEN" : f"OPEN_{k}",
                                "RANGE" : f"RANGE_{k}",
                                "TRADECOUNT" : f"TRADECOUNT_{k}",
                                }, inplace=True)

        df = df[["DATE", "HOUR", f"OPEN_{k}", f"RANGE_{k}", f"VOLUME_{k}", f"TRADECOUNT_{k}"]]

        if full.shape[0] == 0:
            full = df
        else:
            full = pd.merge(full, df, on=["DATE", "HOUR"], how="left", validate="1:1")
    

    # add date variables 
    full["WEEK_DAY"] = full["DATE"].dt.dayofweek
    full["MONTH"] = full["DATE"].dt.month

    return full


def daily_aggregate(full):
    agg = full.drop("HOUR", axis=1).groupby("DATE").sum().reset_index()