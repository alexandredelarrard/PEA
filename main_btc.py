import pandas as pd 
import yaml
from pathlib import Path as pl

from load_data.download_cryptos import load_share_price_data
from data_prep.clean_data_share_price import clean_share_price
from modelling.btc_modelling_classification import main_btc_training
from data_analysis.analyse_share_price import prepare_currency_daily

def load_configs(config_yml):
    config_path = "./configs"
    config_yaml = open(pl(config_path) / pl(config_yml), encoding="utf-8")
    configs = yaml.load(config_yaml, Loader=yaml.FullLoader)
    return configs


if __name__ == "__main__" : 
    configs_model = load_configs("configs_btc_model.yml")
    configs_general = load_configs("configs.yml")

    datas= load_share_price_data(configs_general)
    full = clean_share_price(datas)

    agg_btc = prepare_currency_daily(full, currency="BTC")
    agg_eth = prepare_currency_daily(full, currency="ETH")

    agg = pd.merge(agg_btc, agg_eth, on=["DATE"], how="left", validate="1:1", suffixes=("_BTC", "_ETH"))
    main_btc_training(agg, configs_model)

    # for crypto in ["BTC", "ETH"]:#configs_general["cryptos_desc"]["Cryptos"]:
    #     print("*"*40 + crypto + "*"*40)
    #     agg = prepare_currency_daily(full, currency=crypto)
    #     main_btc_training(agg, configs_model)
    #     print("*"*40 + "*"*40)