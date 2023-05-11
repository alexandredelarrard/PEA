import pandas as pd 
from pathlib import Path as pl
import matplotlib.pyplot as plt

from modelling.picking.stocks.strat3_oecd import load_macroeconomy_data
from modelling.picking.stocks.stocks_data_prep import stock_analysis, encode_features, neutral_market_sector
from modelling.picking.stocks.commodities_prep import add_commodities, add_external_data, concatenate_comos
from modelling.utils.modelling_lightgbm import TrainModel
import joblib
import shap


map_sectors = {"CONSUMER" : ["Consumer Staples", "Utilities"], 
                "FINANCE" : ["Financial Services", "Banks", "Insurance", "Asset Management" , "Financials", "Real Estate"],
                "HEALTH" : ["Health Care Services", "Health Care Equipment", "Biotechnology"], 
                "ENERGY" : ["Oil & Gas", "Energy"],
                "INDUSTRY": ["Industrial Machinery", "Industrials"], 
                "MATERIAL" : ["Chemicals", "Materials", "Construction and Materials"],
                "ELECTRONIC" : ["Semiconductors", "Electronic Components"],
                "TRANSPORT" : ["Aerospace & Defense", "Airlines", "Freight", "Automobiles and Parts"],
                "LEISURE" : ["Hotels & Restaurants", "Apparel, Accessories & Luxury Goods", "Vinters & Tobacco"],
                "SERVICES" : ["Information Services", "Communication Services"],
                "TELCO" : ["Telecommunications", "Application Software"]}


def modelling_price(stocks, configs_general):

    target = configs_general["stock_modelling"]["TARGET"]

    # remove those to predict & very strange
    stocks = stocks.loc[stocks[target].between(-30,30)] # only 
    stocks[target] = stocks[target].clip(-30,30)

    # train 1 model per sector, since commo & macro kpi depends on the business types strongly
    model_sector = {}
    results_sector = {}
    
    for key, sectors in map_sectors.items():

        stocks_sector = stocks.loc[stocks["SECTOR"].isin(sectors)]
        del stocks_sector["Country"]
        stocks_sector.columns = [x.upper() for x in stocks_sector.columns]
        Train_lgb = TrainModel(configs_general["stock_modelling"], data = stocks_sector)
        results, model = Train_lgb.modelling_cross_validation(data=stocks_sector)

        print(f"RESULTS DONE FOR SECTOR {key}")

        model_sector[key] = model
        results_sector[key] = results
 
    return results_sector, model_sector 


def accuracy_curve(results, configs_general): 

    target = configs_general["stock_modelling"]["TARGET"]

    # accuracy + -
    results["PRED_+"] = 1*(results["PREDICTION_" + target ] > 0) 
    results["IS_+"] = 1*(results[target] > 0) 

    right_prediction_sg = (results["IS_+"] == results["PRED_+"])
    x = []
    y=[]

    for value in range(0,21):
        value = value / 100
        x.append(value)

        is_higher_than = (results[target] > value) #| (results["TARGET_STOCK_+5D"] < -1*value)
        response = results.loc[right_prediction_sg & is_higher_than].shape[0]/results[is_higher_than].shape[0]
        y.append(response)

    sub_results = results.loc[results["DATE"] >= "2022-03-01"]
    sub_is_higher_than = (sub_results[target] > 0.05) 
    sub_right_prediction_sg = (sub_results["IS_+"] == sub_results["PRED_+"])
    response = sub_results.loc[sub_right_prediction_sg & sub_is_higher_than].shape[0]/sub_results[sub_is_higher_than].shape[0]
    print(f'ERROR since 01/03 {response}')

    plt.plot(x, y)
    plt.show()


def model_analysis(results_sector, configs_general):

    plt.rcParams["figure.figsize"] = (14, 10)
    import shap

    for key, values in results_sector.items():
        print(key)
        accuracy_curve(values, configs_general)

        valid_x = values.loc[values["DATE"] >= "2021-01-01"][configs_general["stock_modelling"]["FEATURES"]]
        shap_values = shap.TreeExplainer(model_sector[key]).shap_values(valid_x)
        shap.summary_plot(shap_values, valid_x)
        plt.show()

    shap.dependence_plot(
        "SECTOR_DISTANCE_STOCK_0D_TO_1W_MEAN",
        shap_values,
        valid_x
    )

    return shap_values, valid_x


def save_models(model_sector, configs_general):
    for key, value in model_sector.items():
        joblib.dump(value, configs_general["resources"]["base_path"] / pl(f"data/results/model_stocks/{key}.pkl"))


def to_predict(sub_stocks, configs_general):

    features_list = configs_general["stock_modelling"]["FEATURES"]
    # sub_stocks = stocks.loc[stocks["Date"] == stocks["Date"].max()]

    full_data = pd.DataFrame()
    
    for key, sectors in map_sectors.items():
        stocks_sector = sub_stocks.loc[sub_stocks["SECTOR"].isin(sectors)]
        del stocks_sector["Country"]
        stocks_sector.columns = [x.upper() for x in stocks_sector.columns]
        model = joblib.load(configs_general["resources"]["base_path"] / pl(f"data/results/model_stocks/{key}.pkl"))
        
        stocks_sector["PREDICTION"] = model.predict_proba(stocks_sector[features_list],
                                                    categorical_feature=configs_general["stock_modelling"]["categorical_features"])[:, 1]
        results_sector = stocks_sector[["PREDICTION", "DATE", "STOCK", "COMPANY", "SECTOR", "DISTANCE_STOCK_0D_TO_1W_MEAN", "DISTANCE_STOCK_0D_TO_4W_MEAN", "DISTANCE_STOCK_0D_TO_8W_MEAN", "DISTANCE_VOLUME_0D_TO_4W_MEAN"]]
        full_data = pd.concat([full_data, results_sector], axis=0)

    return full_data


if __name__ == "__main__":

    mode = "train"
    configs_general = load_configs("configs_pea.yml")
    target = configs_general["stock_modelling"]["TARGET"]

    dict_macroeconomy_data = load_macroeconomy_data(configs_general["resources"]["base_path"])
    dict_macroeconomy_data["COMPANIES"] = pd.read_csv(configs_general["resources"]["base_path"] / pl("data/data_for_crawling/mapping_reuters_yahoo.csv"), sep=";")
    dict_macroeconomy_data["COMPANIES"]["SECTOR"] = dict_macroeconomy_data["COMPANIES"]["SECTOR"].apply(lambda x: x.strip())
    full_commos = concatenate_comos(dict_macroeconomy_data)

    stocks, missing = stock_analysis(configs_general, dict_macroeconomy_data["COMPANIES"], full_commos)
    stocks = add_commodities(stocks, dict_macroeconomy_data)
    stocks = add_external_data(stocks, dict_macroeconomy_data)
    stocks = encode_features(stocks)
    stocks = neutral_market_sector(stocks)

    predictions = stocks.loc[stocks["Date"] >= pd.to_datetime("2022-06-26", format="%Y-%m-%d")]
    stocks = stocks.loc[stocks["TARGET_STOCK_+5D"].between(-1,1)]
    stocks[target] = (stocks["TARGET_STOCK_+5D"] > 0)*1

    # modelling
    if mode == "train":
        results_sector_train, model_sector  = modelling_price(stocks, configs_general)
        # shap_values, valid_x = model_analysis(results_sector_train, configs_general)
        save_models(model_sector, configs_general)
 
    # prediction_next week
    if mode == "test":
        results_sector = to_predict(predictions, configs_general)
        # high = results_sector.loc[(results_sector["PREDICTION" + target] > 0.07)]
        # high = high.sort_values("PREDICTION"+ target)[-60:]
        # low = results_sector.loc[(results_sector["PREDICTION"+target] < -0.07)]
        # low = low.sort_values("PREDICTION"+ target)[:60]