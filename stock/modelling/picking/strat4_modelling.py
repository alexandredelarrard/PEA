import pandas as pd 
import seaborn as sns
import numpy as np
import scipy.stats as ss

from modelling.utils.modelling_lightgbm import TrainModel

def post_data_prep(final):

    df = final.copy() 

    df["PROFILE_RATING"].fillna(df["PROFILE_RATING"].median(), inplace=True)
    df["PROFILE_RATING"] = np.where(df["PROFILE_RATING_NBR_ANALYSTS"] <= 5, df["PROFILE_RATING"].median(), df["PROFILE_RATING"])
    df["PROFILE_RATING"] = 6 - df["PROFILE_RATING"]

    # pre clean data 
    for i in range(4):

        # P/E
        for col in ["_P/E_Y-", "DISTANCE MARKET_CAP / INTRINSIC _", "_P/E_+1Q_Y-",]:
            df[f"{col}{i}"] = np.where(df[f"{col}{i}"] < 0, 1000, 
                            np.where(df[f"{col}{i}"] > 1000, 1000, df[f"{col}{i}"]))
            df[f"{col}{i}"] = df[f"{col}{i}"].clip(0.5, 1000)

        # _NET_PROFIT_MARGIN_
        for col in ["%_NET_PROFIT_MARGIN", '%_R&D_IN_OPERATING_INCOME', '%_SALES_GENERAL_IN_REVENUE',
                    '%_TAXES_&_EXTRA_IN_TOTAL_REVENUE', 'CASH_%_DIVIDENDS IN ACTIVITY_CASH',
                    'CASH_%_FINANCE IN ACTIVITY_CASH', 'CASH_%_INTO_ACQUISITION', 
                    'CASH_%_INTO_ACTIVITY', 'BALANCE_%_SHARE_GOODWILL_ASSETS']:
            if f"{col}_{i}" in df.columns:
                df[f"{col}_{i}"] = df[f"{col}_{i}"].clip(0, 100)

        for col in ['%_YO2Y_NET_INCOME_BEFORE_EXTRA_ITEMS',
                    '%_YO2Y_OPERATING_INCOME',
                    '%_YO2Y_TOTAL_REVENUE',
                    '%_YOY_NET_INCOME_BEFORE_EXTRA_ITEMS',
                    '%_YOY_OPERATING_INCOME',
                    '%_YOY_R&D',
                    '%_YOY_TOTAL_REVENUE',
                    'CASH_TREND_FROM_OPERATING_ACTIVITIES',
                    'TREND_OP_INCOME / TREND_TOTAL_REVENUE',
                    'STOCK_LONG_TREND_CLOSE_PLUS_1Q',
                    'LONG_TREND_NET_INCOME_BEFORE_EXTRA_ITEMS',
                    'LONG_TREND_OP_INCOME',
                    'LONG_TREND_R&D_SHARE_OPERATING_INCOME',
                    'LONG_TREND_SHARE_SALES_GENERAL_IN_REVENUE',
                    'LONG_TREND_TOTAL_REVENUE',
                    'BALANCE_TREND_SHAREHOLDERS_EQUITY',
                    'BALANCE_TREND_CURRENT_ASSET / CURRENT_DEBT']:
            if f"{col}_{i}" in df.columns:
                df[f"{col}_{i}"] = df[f"{col}_{i}"].clip(-250, 250)

        for col in ['BALANCE_%_CASH / TOTAL_DEBT',
                    'BALANCE_%_CURRENT_ASSET / CURRENT_DEBT',
                    'BALANCE_%_MARKET_CAP / SHAREHOLDERS_EQUITY',
                    'BALANCE_%_TOTAL_ASSET / TOTAL_LIABILITIES']:
            if f"{col}_{i}" in df.columns:
                df[f"{col}_{i}"] = df[f"{col}_{i}"].clip(0, 5000)

    # add distance to neighbors features 
    for col in ["%_NET_PROFIT_MARGIN", '%_YOY_TOTAL_REVENUE',
                '%_YOY_NET_INCOME_BEFORE_EXTRA_ITEMS', '%_YOY_OPERATING_INCOME',
                'BALANCE_%_CURRENT_ASSET / CURRENT_DEBT',  'CASH_%_INTO_ACQUISITION',
                'CASH_%_FINANCE IN ACTIVITY_CASH', 'DISTANCE MARKET_CAP / INTRINSIC ']:
        df[f"AVG_{col}"] = df[[f'{col}_0',
                                f'{col}_1',
                                f'{col}_2']].mean(axis=1)

    df =  deduce_ranking(df, ["%_NET_PROFIT_MARGIN_0", 
                            "%_NET_PROFIT_MARGIN_1", 
                            "%_NET_PROFIT_MARGIN_2", 
                            '%_YOY_TOTAL_REVENUE_0',
                            '%_YOY_TOTAL_REVENUE_1',
                            '%_YOY_TOTAL_REVENUE_2',
                            '%_YOY_NET_INCOME_BEFORE_EXTRA_ITEMS_0',
                            '%_YOY_NET_INCOME_BEFORE_EXTRA_ITEMS_1',
                            '%_YOY_NET_INCOME_BEFORE_EXTRA_ITEMS_2',
                            'BALANCE_%_SHARE_GOODWILL_ASSETS_0',
                            'BALANCE_%_SHARE_GOODWILL_ASSETS_1',
                            'BALANCE_%_SHARE_GOODWILL_ASSETS_2',
                            'STOCK_%_52_WEEKS_0',
                            'STOCK_%_52_WEEKS_1',
                            'STOCK_%_52_WEEKS_2',
                            '_P/E_Y-0',
                            '_P/E_Y-1',
                            '_P/E_Y-2',
                            '_P/E_Y-3',
                            'BALANCE_TREND_CURRENT_ASSET / CURRENT_DEBT_0',
                            'BALANCE_TREND_CURRENT_ASSET / CURRENT_DEBT_1',
                            'BALANCE_TREND_CURRENT_ASSET / CURRENT_DEBT_2'])

    return df


def deduce_ranking(df, vars):

    negihbors_deduced = pd.DataFrame(index=df.index)

    for var in vars:
        negihbors_deduced["MEAN_" + var] = np.nan
        negihbors_deduced["%_DISTANCE_" + var] = np.nan
        negihbors_deduced["RANKING_" + var] = np.nan

    for company in df.index:
        sub_neighbors = df.loc[company]
        weights = np.array(df.loc[company, "WEIGHTS"])
        weights = weights[weights >= 0.2]
        top = len(weights)

        if top <=1:
            print(company)

        if top > 1:
            sub_df = df.loc[sub_neighbors["NEIGHBORS"][:top]]

            for var in vars:
                negihbors_deduced.loc[company, "MEAN_" + var] = sum(sub_df[var]*weights)/sum(weights)
                negihbors_deduced.loc[company, "%_DISTANCE_" + var] = (sub_neighbors[var] - negihbors_deduced.loc[company, "MEAN_" + var]) / negihbors_deduced.loc[company, "MEAN_" + var]
                negihbors_deduced.loc[company, "RANKING_" + var] = top + 2 - ss.rankdata([sub_neighbors[var]] + list(sub_df[var].values))[0]

                negihbors_deduced.loc[company, "%_DISTANCE_" + var] = negihbors_deduced.loc[company, "%_DISTANCE_" + var].clip(-1, 10)

    df = pd.merge(df, negihbors_deduced, left_index=True, right_index=True, how="left", validate="1:1")

    return df


def modelling_pe(df, configs_general):

    df = df[[x for x in df.columns if 'STOCK_LONG_TREND_CLOSE_PLUS' not in x]]
    df.columns = [x.upper() for x in df.columns]

    # build training / testing data 
    common_cols = [
                    "SECTOR", 'TEAM_#_INDEPENDENT_DIRECTOR', 'TEAM_LEADER_AGE_AVG', 
                    'PROFILE_RATING',  'PROFILE_RATING_NBR_ANALYSTS',
                    'TEAM_CEO_APPOINTED',
                    'TEAM_C_LEVEL_AVG_APPOINTED',
                    'TEAM_LEADER_APPOINTED_AVG',
                    'COUNTRY',
                    "SPECIFIC",
                    "NAME", 
                    "PROFILE_DESC",
                    'SUB INDUSTRY'
                    ]

    target = configs_general["regression_model"]["TARGET"]

    columns_0 = [x for x in df.columns if "_0" in x or "-0" in x] + common_cols + ['_P/E_Y-1', 'MEAN__P/E_Y-1', '%_DISTANCE__P/E_Y-1', 'RANKING__P/E_Y-1', '%_YOY_NET_INCOME_BEFORE_EXTRA_ITEMS_1']
    columns_1 = [x for x in df.columns if "_1" in x or "-1" in x] + common_cols + ['_P/E_Y-2', 'MEAN__P/E_Y-2', '%_DISTANCE__P/E_Y-2', 'RANKING__P/E_Y-2', '%_YOY_NET_INCOME_BEFORE_EXTRA_ITEMS_2']
    columns_2 = [x for x in df.columns if "_2" in x or "-2" in x] + common_cols + ['_P/E_Y-3', 'MEAN__P/E_Y-3', '%_DISTANCE__P/E_Y-3', 'RANKING__P/E_Y-3', '%_YOY_NET_INCOME_BEFORE_EXTRA_ITEMS_3']

    df0 = df[columns_0]      
    df1 = df[columns_1]   
    df2 = df[columns_2]   

    df1.columns = [x.replace("_1","").replace("-1","") for x in df1.columns]
    df1.columns = [x.replace("_2","-1").replace("-2","-1") for x in df1.columns]
    df2.columns = [x.replace("_2","").replace("-2","") for x in df2.columns]
    df2.columns = [x.replace("_3","-1").replace("-3","-1") for x in df2.columns]

    # training data 
    training = pd.concat([df1, df2], axis=0)
    training = training.reset_index()
    training["%_P/E_Y"] = (training["_P/E_Y"] - training["_P/E_Y-1"])*100/(training["_P/E_Y-1"])
    training["DELTA_P/E_Y"] = (training["_P/E_Y"] - training["_P/E_Y-1"])
    training = training.loc[~training[target].isnull()]
    
    # testing_data
    df0.columns = [x.replace("_0","").replace("-0","") for x in df0.columns]
    df0.columns = [x.replace("_1","-1") for x in df0.columns]

    testing = df0
    testing = testing.reset_index()
    testing["%_P/E_Y"] = (testing["_P/E_Y"] - testing["_P/E_Y-1"])*100/(testing["_P/E_Y-1"])
    testing["DELTA_P/E_Y"] = (testing["_P/E_Y"] - testing["_P/E_Y-1"])
    testing = testing.loc[~testing[target].isnull()]
    
    # transform to right data types 
    training_cols = configs_general["regression_model"]["FEATURES"]
    for col in training_cols:
        if col not in configs_general["regression_model"]["categorical_features"]:
            training[col] =  training[col].astype("float")
            testing[col] = testing[col].astype("float")
        else:
            training[col] = training[col].astype("category")
            testing[col] = testing[col].astype("category")

    training[configs_general["regression_model"]["TARGET"]] = training[configs_general["regression_model"]["TARGET"]].astype(float)
    testing[configs_general["regression_model"]["TARGET"]] = testing[configs_general["regression_model"]["TARGET"]].astype(float)

    if configs_general["regression_model"]["TARGET"] == "DELTA_P/E_Y":
        training = training.loc[training["DELTA_P/E_Y"].between(-50,50)]
        testing = testing.loc[testing["DELTA_P/E_Y"].between(-50,50)]
    else:
        training = training.loc[training[configs_general["regression_model"]["TARGET"]] < 200]
        training = training.loc[training["_P/E_Y-1"] != 1000]

    Train_lgb = TrainModel(configs_general["regression_model"], data = training)
    results, models = Train_lgb.modelling_cross_validation(data=training)

    # sector error 
    a = results[["_P/E_+1Q_Y", "_P/E_Y", "_P/E_Y-1", "PREDICTION__P/E_+1Q_Y", "%_YOY_NET_INCOME_BEFORE_EXTRA_ITEMS", '%_NET_PROFIT_MARGIN', "SECTOR", "NAME"]]
    a["%_ERROR_SECTOR"] = np.abs((results["_P/E_+1Q_Y"] - results["PREDICTION__P/E_+1Q_Y"])*100/results["_P/E_+1Q_Y"])
    b = a[["SECTOR", "%_ERROR_SECTOR"]].groupby("SECTOR").mean().sort_values("%_ERROR_SECTOR")

    testing["P/E_PREDICTION"] = models.predict(testing[training_cols])
    testing.index = testing["index"]
    testing = pd.merge(testing, b, left_on="SECTOR", right_index=True, how="left")

    return results, models, testing[["P/E_PREDICTION", "%_ERROR_SECTOR"]]


# if __name__ == "__main__":

#     a = results[["_P/E_Y", "_P/E_Y-1", "PREDICTION__P/E_Y", "%_YOY_NET_INCOME_BEFORE_EXTRA_ITEMS", '%_NET_PROFIT_MARGIN', "SECTOR", "NAME"]]
#     a["DIFF"] = np.abs((results["_P/E_Y"] - results["PREDICTION__P/E_Y"])*100/results["_P/E_Y"])

#     a[["SECTOR", "DIFF"]].groupby("SECTOR").mean().sort_values("DIFF")
#     sns.scatterplot(x="_P/E_Y", y="DIFF", data=a)

    # import shap 
    # X = training[configs_general["regression_model"]["FEATURES"]]
    # shap_values = shap.TreeExplainer(models).shap_values(X)
    # shap.summary_plot(shap_values, X)

    # for col in configs_general["regression_model"]["FEATURES"]:
    #     shap.dependence_plot(col, shap_values, X)
