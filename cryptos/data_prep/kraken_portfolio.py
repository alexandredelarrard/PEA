
from kraken.spot import User, Market, Trade, Funding, Staking
import pandas as pd 
import pickle
import glob
from datetime import datetime
import os

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

    def get_past_trades(self, prepared):

        trades  = self.user.get_ledgers_info()
        df_trades = pd.DataFrame.from_dict(trades["ledger"]).T

        df_trades["time"] = pd.to_datetime(df_trades["time"], unit='s')
        df_trades["time"] = df_trades["time"].dt.round("H")

        # buy / sell 
        receive = df_trades.loc[df_trades["type"] == "receive"][["refid", "time", "asset", "amount", "balance"]]
        spend = df_trades.loc[df_trades["type"] == "spend"][["refid", "amount", "fee"]]
        spend = spend.rename(columns={"amount" : "cost/gain"})

        receive = receive.merge(spend, on=["refid"], how="left", validate="1:1")
        receive["asset"] = receive["asset"].map(self.mapping_kraken)
        # receive["fee"] = ((receive["fee"].astype(float)/receive["cost/gain"].astype(float).abs())*100).round(1)

        for k, v in {"amount": 4, "balance":4, "cost/gain":2, "fee":1}.items():
            receive[k] = receive[k].astype(float).round(v)

        #  get +/ - value 
        get_price = prepared.iloc[0:2]
        receive["CURRENT_VALUE"] = 0
        receive["MARGIN_EUR"] = 0
        receive["MARGIN_%"] = 0
        for currency in self.currencies:
            condition = receive["asset"] == currency
            if_sold_price = receive.loc[condition, "amount"]*get_price["CLOSE_" + currency].bfill()[0]
            receive.loc[condition, "MARGIN_EUR"] = (if_sold_price*(1-0.026) + 
                                                    receive.loc[condition, "cost/gain"] - receive.loc[condition, "fee"]).round(3)

            receive.loc[condition, "MARGIN_%"] = (100*receive.loc[condition, "MARGIN_EUR"] / (receive.loc[condition, "cost/gain"].abs() + receive.loc[condition, "fee"])).round(3)
            receive.loc[condition, "CURRENT_VALUE"] = if_sold_price.round(3)

        return receive.drop(["refid"], axis=1)
        

    def pnl_over_time(self, receive, prepared):

        pnl_prepared = prepared[["DATE"]+[f"CLOSE_{x}" for x in self.currencies]]
        pnl_prepared = pnl_prepared.iloc[1:]
        pnl_prepared = pnl_prepared.sort_values("DATE", ascending= 1)
        pnl_prepared["CASH"] = 8000

        for currency in self.currencies:
            condition = receive["asset"] == currency
            cols = ["time", "amount", "fee", "cost/gain"]
            pnl_prepared = pnl_prepared.merge(receive.loc[condition][cols], left_on="DATE", right_on="time", how="left", validate="1:1")
            pnl_prepared["amount"] = pnl_prepared["amount"].fillna(0).cumsum()
            pnl_prepared["cost/gain"] = pnl_prepared["cost/gain"].fillna(0).cumsum()
            pnl_prepared["CASH"] += pnl_prepared["cost/gain"]
            pnl_prepared[f"PNL_{currency}"] = pnl_prepared[f"CLOSE_{currency}"]*pnl_prepared["amount"]
            
            pnl_prepared = pnl_prepared.drop(["time", "amount", f"CLOSE_{currency}", "fee", "cost/gain"], axis=1)

        pnl_prepared["PNL_PORTFOLIO"] = pnl_prepared.sum(axis=1, numeric_only=True)

        return pnl_prepared
    
    #### SAVERS / LOADERS
    def save_trades(self, trades):
        utcnow = datetime.today().strftime("%Y-%m-%d_%H-%S")
        pickle.dump(trades, open("/".join([self.path_dirs["PORTFOLIO"], f"trades_{utcnow}.pkl"]), 'wb'))

    def save_pnl(self, pnl):
        utcnow = datetime.today().strftime("%Y-%m-%d_%H-%S")
        pickle.dump(pnl, open("/".join([self.path_dirs["PORTFOLIO"], f"pnl_{utcnow}.pkl"]), 'wb'))

    def save_df_init(self, df_init):
        utcnow = datetime.today().strftime("%Y-%m-%d_%H-%S")
        pickle.dump(df_init, open("/".join([self.path_dirs["PORTFOLIO"], f"df_init_{utcnow}.pkl"]), 'wb'))

    def save_orders(self, orders):
        utcnow = datetime.today().strftime("%Y-%m-%d_%H-%S")
        pickle.dump(orders, open("/".join([self.path_dirs["ORDERS"], f"orders_{utcnow}.pkl"]), 'wb'))

    def load_trades(self):
        list_of_files = glob.glob(self.path_dirs["PORTFOLIO"]+"/trades_*") 
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
            return df_orders
        else:
            return None
    