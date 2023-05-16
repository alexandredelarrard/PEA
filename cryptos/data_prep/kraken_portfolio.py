
from kraken.spot import User, Market, Trade
import pandas as pd 
import numpy as np
import pickle
import glob
from datetime import datetime
import os

from utils.general_cleaning import smart_column_parser

class OrderKraken(object):

    def __init__(self, configs, paths):

        self.key = os.environ["API_KRAKEN"]
        self.secret = os.environ["API_PRIVATE_KRAKEN"]

        self.configs = configs
        self.path_dirs=paths
        self.currencies = self.configs.load["cryptos_desc"]["Cryptos"]
        self.mapping_kraken = {"XETH" : "ETH",
                                "XXBT" : "BTC",
                                "XXLM" : "XLM",
                                "XXRP" : "XRP",
                                "ZEUR" : "CASH", 
                                "STX" : "STX",
                                "ADA" : "ADA",
                                "ICP" : "ICP",
                                "SOL" : "SOL",
                                "TRX" : "TRX",
                                "SAND" : "SAND"}
        self.init_cash = 8000
        
        self.auth_trade = Trade(key=self.key, secret=self.secret)
        self.user = User(key=self.key, secret=self.secret)

    def get_current_portolio(self):

        infos = pd.DataFrame.from_dict({"BALANCE" : self.user.get_account_balance()}).T 
        infos.rename(columns = self.mapping_kraken, inplace=True)
        
        remaining_currencies = [x for x in self.currencies if x not in infos.columns]
        for col in remaining_currencies:
            infos[col] = 0

        return infos
    
    def cancel_orders(self, seconds=90):
        nbr_orders = self.auth_trade.cancel_all_orders_after_x(timeout=seconds)
        return nbr_orders
    
    def get_latest_price(self):

        latest_price = {}
        for currency in self.currencies:
            price = Market().get_ohlc(pair=f"{currency}EUR")
            latest_price[currency] = float(price[next(iter(price))][-1][4])
        
        return latest_price

    def get_open_orders(self):
        open_trades = self.user.get_open_orders()["open"]

        df_orders = []
        for k, order in open_trades.items():
            df_orders.append(order["descr"])

        return pd.DataFrame(df_orders)
    
    def get_closed_orders(self):
        closed_trades = self.user.get_closed_orders()["closed"]

        df_orders = []
        for k, order in closed_trades.items():
            answer = order["descr"]
            answer["vol_exec"] = order["vol_exec"]
            answer["status"] =  order["status"]
            answer["opentm"] =  datetime.fromtimestamp(order["opentm"])
            answer["closetm"] =  datetime.fromtimestamp(order["closetm"])

            df_orders.append(answer)

        return pd.DataFrame(df_orders)


    def get_past_trades(self):

        trades = self.user.get_ledgers_info()
        df_trades = pd.DataFrame.from_dict(trades["ledger"]).T

        df_trades["time"] = pd.to_datetime(df_trades["time"], unit='s')
        df_trades["time"] = df_trades["time"].dt.round("S")

        for col in ["amount", "fee", "balance"]:
            df_trades[col] = df_trades[col].astype(float)

        # extract spend in eur 
        bought = df_trades.loc[(df_trades["amount"] < 0)&(df_trades["asset"] == "ZEUR")]
        counterpart = df_trades.loc[(~df_trades.index.isin(bought.index))&(df_trades["asset"] !="ZEUR")][["refid", "asset", "amount", "fee"]]
        bought = bought.merge(counterpart, on="refid", how="left", validate="1:1", suffixes=("_CASH", "_COIN"))
        bought["COIN_PRICE"] = bought["amount_CASH"].abs()/bought["amount_COIN"]
        bought = bought[["time", "asset_COIN", "amount_COIN", "amount_CASH", "fee_CASH", "COIN_PRICE"]]
        bought["COIN_PRICE_SELL"] = np.nan
        bought["EARNED_MARGIN"] = np.nan
        bought["fee_SOLD"] = np.nan
        bought["amount_COIN_SOLD"] = np.nan
        bought["time_sold"] = np.nan
        bought["TO_EARN_MARGIN"] = 0
        bought["NET_%_MARGIN"] = 0
        
        # sell
        sold = df_trades.loc[(df_trades["amount"] < 0)&(df_trades["asset"] != "ZEUR")]
        counterpart = df_trades.loc[(~df_trades.index.isin(sold.index))&(df_trades["asset"] =="ZEUR")][["refid", "asset", "amount", "fee"]]
        sold = sold.merge(counterpart, on="refid", how="left", validate="1:1", suffixes=("_COIN", "_CASH"))
        sold["COIN_PRICE"] = sold["amount_CASH"]/sold["amount_COIN"].abs()
        sold["fee"] = sold["fee_COIN"]*sold["COIN_PRICE"] + sold["fee_CASH"]
        sold = sold[["time", "asset_COIN", "amount_COIN", "amount_CASH", "fee", "COIN_PRICE"]]
        sold = sold.sort_values("time", ascending=False)

        for index, row in sold.iterrows():
            sub_bought = bought.loc[(bought["time"] <= row["time"])&(bought["asset_COIN"] == row["asset_COIN"])]
            sub_bought = sub_bought.sort_values("time", ascending=True)
            remaining_coins_to_sell = abs(row["amount_COIN"])
            for sub_index, sub_row in sub_bought.iterrows():
                if sub_row["amount_COIN"]>0:
                    final_amont_coins = max(0, sub_row["amount_COIN"] - remaining_coins_to_sell)
                    remaining_coins_to_sell -= sub_row["amount_COIN"]
                    nbr_sold_coins =  (sub_row["amount_COIN"] - final_amont_coins)
                    bought.loc[sub_index, "amount_COIN"] = final_amont_coins
                    bought.loc[sub_index, "EARNED_MARGIN"] = nbr_sold_coins*(row["COIN_PRICE"] - sub_row["COIN_PRICE"])
                    bought.loc[sub_index, "COIN_PRICE_SELL"] = row["COIN_PRICE"]
                    bought.loc[sub_index, "fee_SOLD"] = row["fee"]*(nbr_sold_coins/abs(row["amount_COIN"]))
                    bought.loc[sub_index, "amount_COIN_SOLD"] = nbr_sold_coins
                    bought.loc[sub_index, "time_sold"] = row["time"]

        bought["asset_COIN"] = bought["asset_COIN"].map(self.mapping_kraken)
        bought["CURRENT_COIN_PRICE"] = bought["asset_COIN"].map(self.get_latest_price())
        bought["TO_EARN_MARGIN"] = (bought["CURRENT_COIN_PRICE"] - bought["COIN_PRICE"])*bought["amount_COIN"]
        bought["amount_CASH"] = bought["amount_CASH"].abs()
        bought["NET_MARGIN_PERCENT"] = (100*(bought["EARNED_MARGIN"].fillna(0) + bought["TO_EARN_MARGIN"] - bought["fee_SOLD"].fillna(0) - bought["fee_CASH"]) / bought["amount_CASH"]).round(2)

        bought = bought[["time", "time_sold", "asset_COIN", "amount_CASH", "amount_COIN", "COIN_PRICE", "COIN_PRICE_SELL", "CURRENT_COIN_PRICE", 
                         "amount_COIN_SOLD", "fee_CASH", "fee_SOLD", "EARNED_MARGIN", "TO_EARN_MARGIN", "NET_MARGIN_PERCENT"]]
        
        bought.rename(columns={"time" : "time_buy", "asset_COIN" : "asset", "amount_CASH" : "amount_eur", 
                               "amount_COIN" : "volume_coin", "COIN_PRICE" : "price_buy", "COIN_PRICE_SELL" : "price_sell",  "CURRENT_COIN_PRICE" : "current_price",
                         "amount_COIN_SOLD" : "volume_coin_sell", "fee_CASH" : "fee_buy", "fee_SOLD" : "fee_sell"}, inplace=True)
        bought.columns = smart_column_parser(bought.columns)
        bought["TOTAL_FEE"] = bought["FEE_BUY"] + bought["FEE_SELL"]

        #for portfolio overall
        overall = (bought[["ASSET", "VOLUME_COIN", "CURRENT_PRICE", "TOTAL_FEE", "EARNED_MARGIN", "TO_EARN_MARGIN", "NET_MARGIN_PERCENT"]]
                   .groupby("ASSET")
                   .aggregate({
                            "VOLUME_COIN" : sum,
                            "CURRENT_PRICE" : np.mean,
                            "EARNED_MARGIN" : sum,
                            "TO_EARN_MARGIN" : sum,
                            "TOTAL_FEE" : sum,
                            "NET_MARGIN_PERCENT" : np.mean,
                            })
        )

        bought["TIME_BUY"] = bought["TIME_BUY"].dt.round("H")

        return bought, overall
        

    def pnl_over_time(self, bought):

        bought = bought.copy()
        
        now= datetime.today()
        bought["TIME_SOLD"] = bought["TIME_SOLD"].fillna(now).dt.round("H")
        bought = bought.sort_values("TIME_SOLD", ascending = True)
        bought["FEE_SELL"].fillna(0, inplace=True)
        bought["EARNED_MARGIN"].fillna(0, inplace=True)
        bought["FEE_BUY"].fillna(0, inplace=True)

        dates = pd.date_range(bought["TIME_SOLD"].min(), bought["TIME_SOLD"].max(), freq="h")
        pnl_prepared = pd.DataFrame(dates, columns=["DATE"])
        pnl_prepared["PNL_PORTFOLIO"] = self.init_cash

        for index, row in bought.iterrows():
            condition = pnl_prepared["DATE"].between(row["TIME_SOLD"], now)
            pnl_prepared.loc[condition, "PNL_PORTFOLIO"] += row["EARNED_MARGIN"] + row["TO_EARN_MARGIN"] - row["FEE_SELL"] - row["FEE_BUY"]

        return pnl_prepared
    
    #### SAVERS / LOADERS
    def save_trades(self, trades):
        utcnow = datetime.today().strftime("%Y-%m-%d_%H-%M")
        pickle.dump(trades, open("/".join([self.path_dirs["PORTFOLIO"], f"trades_{utcnow}.pkl"]), 'wb'))

    def save_global_portfolio(self, portfolio):
        utcnow = datetime.today().strftime("%Y-%m-%d_%H-%M")
        pickle.dump(portfolio, open("/".join([self.path_dirs["PORTFOLIO"], f"portfolio_{utcnow}.pkl"]), 'wb'))

    def save_pnl(self, pnl):
        utcnow = datetime.today().strftime("%Y-%m-%d_%H-%M")
        pickle.dump(pnl, open("/".join([self.path_dirs["PORTFOLIO"], f"pnl_{utcnow}.pkl"]), 'wb'))

    def save_df_init(self, df_init):
        utcnow = datetime.today().strftime("%Y-%m-%d_%H-%M")
        pickle.dump(df_init, open("/".join([self.path_dirs["PORTFOLIO"], f"df_init_{utcnow}.pkl"]), 'wb'))

    def save_orders(self, orders):
        if orders.shape[0]>0:
            utcnow = datetime.today().strftime("%Y-%m-%d_%H-%M")
            pickle.dump(orders, open("/".join([self.path_dirs["ORDERS"], f"orders_{utcnow}.pkl"]), 'wb'))

    def load_trades(self):
        list_of_files = glob.glob(self.path_dirs["PORTFOLIO"]+"/trades_*") 
        if len(list_of_files)>0:
            latest_file = max(list_of_files, key=os.path.getctime)
            return pickle.load(open(latest_file, 'rb'))
        else:
            return None
        
    def load_global_portfolio(self):
        list_of_files = glob.glob(self.path_dirs["PORTFOLIO"]+"/portfolio_*") 
        if len(list_of_files)>0:
            latest_file = max(list_of_files, key=os.path.getctime)
            return pickle.load(open(latest_file, 'rb'))
        else:
            return None
        
    def load_pnl(self):

        list_of_files = glob.glob(self.path_dirs["PORTFOLIO"]+"/pnl_*") 
        if len(list_of_files)>0:
            latest_file = max(list_of_files, key=os.path.getctime)
            return pickle.load(open(latest_file, 'rb'))
        else:
            return None
    
    def load_df_init(self):

        list_of_files = glob.glob(self.path_dirs["PORTFOLIO"]+"/df_init_*") 
        if len(list_of_files)>0:
            latest_file = max(list_of_files, key=os.path.getctime)
            return pickle.load(open(latest_file, 'rb'))
        else:
            return None
        
    def load_orders(self):

        list_of_files = glob.glob(self.path_dirs["ORDERS"]+"/orders_*") 
        if len(list_of_files)>0:
            df_orders = pd.DataFrame()
            for file in list_of_files: 
                df = pickle.load(open(file, 'rb'))
                df_orders = pd.concat([df_orders, df], axis=0)
            return df_orders.reset_index(drop=True)
        else:
            return None
    