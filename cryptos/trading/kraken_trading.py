
from kraken.spot import User, Market, Trade
import pandas as pd 
import logging
import os
from datetime import datetime
import time
from utils.general_functions import smart_column_parser

class TradingKraken(object):

    def __init__(self, configs):

        self.key = os.environ["API_KRAKEN"]
        self.secret = os.environ["API_PRIVATE_KRAKEN"]

        self.configs = configs
        # self.expire_time_order = 120
        self.currencies = self.configs.load["cryptos_desc"]["Cryptos"]

        self.trade_min_vol = {  "BTC" : 0.0001,
                                "ETH" : 0.01,
                                "ADA" : 15,
                                "XRP" : 15,
                                "XLM" : 60,
                                "STX" : 20,
                                "SOL" : 0.25,
                                "ICP" : 1,
                                "DOGE": 60,
                                "TRX" : 100,
                                "SAND": 8.5
                            }
        self.price_decimals = { 
                                "ICP" : 3,
                                "SAND": 4,  
                                "DOGE": 7,
                                "ADA" : 6,
                                "BTC" : 1,
                                "ETH" : 2,
                                "XRP" : 5,
                                "XLM" : 6,
                                "STX" : 4,
                                "SOL" : 2,
                                "TRX" : 6
                            }

        self.auth_trade = Trade(key=self.key, secret=self.secret)
        self.auth_user = User(key=self.key, secret=self.secret) 
    
    def deduce_side(self, row):
        if row == 1:
            return "buy"
        elif row == -1:
            return "sell"
        else:
            logging.error("REAL_BUY_SELL need to be either -1 or 1")
            return "None"
        
    def deduce_pair(self, row):
        if "EUR" not in row and row in self.currencies:
            return row + "EUR"
        else:
            logging.warning(f"{row} not in {self.currencies}")
            return "None"
        
    def deduce_price(self, row, side, pair, decimals):

        ticker = Market().get_ticker(pair=pair)
        ticker = ticker[next(iter(ticker))]

        if side == "buy":
            ticker_price = float(ticker["b"][0])
            price = round(min(ticker_price, row*0.9975), decimals)
            delta = (ticker_price - price)*100 / ticker_price # si price << ticjer price then ok 

        elif side == "sell":
            ticker_price = float(ticker["a"][0])
            price = round(max(ticker_price, row*1.002), decimals)
            delta = (price - ticker_price)*100 / ticker_price

        else:
            raise Exception("Side can be only buy or sell in orders")
        
        if delta > 1.5:
            comment_price = "NO_PASS"
        else:
            comment_price = "PASS"
        
        return price, ticker_price, comment_price
    

    def deduce_volume(self, row, side, price, nbr_coins):

        if side == "buy":
            volume = round((abs(row)*(1-0.015) / price), 7)
        
        elif side == "sell":
            volume = nbr_coins

        else:
            raise Exception("Side can be only buy or sell in orders")
        
        return volume
    

    def validate_trading_conditions(self, prepared, df_init):

        # ensure latest date is accessible
        now = datetime.utcnow()
        time_diff_to_latest = (now - prepared["DATE"].max()).total_seconds()/60

        if time_diff_to_latest > 15:
            logging.warning("[NOT TRADING] latest data more than 15 minutes ago")
            return False 
        
        if df_init.isnull().sum().sum()>0:
            logging.warning("[NOT TRADING] exists a MV in df_init")
            return False
        
        return True
        

    def validate_orders(self, df_init, moves_prepared):

        # ensure different index per row
        moves_prepared = moves_prepared.reset_index(drop=True)

        orders = {}
        for index, row in moves_prepared.iterrows():

            currency = row["CURRENCY"]

            # get pair 
            pair = self.deduce_pair(currency)
            side = self.deduce_side(row["REAL_BUY_SELL"])

            ### check if price match or not 
            price, tick_price, comment_price = self.deduce_price(row["PRICE"], 
                                                                side, 
                                                                pair, 
                                                                self.price_decimals[currency])

            ## deduce volume 
            nbr_coins = float(df_init.loc["BALANCE", currency])
            volume = self.deduce_volume(row["AMOUNT"], side, price, nbr_coins)

            # all but price / volume 
            min_volume = self.trade_min_vol[currency]
            if comment_price == "PASS" and volume >= min_volume:
                orders[index] = {"ordertype" : "limit",
                                "pair" : pair,
                                "side" : side,
                                "price" : price,
                                "volume" : volume,
                                # "expiretm": self.expire_time_order, # expire au bout de ~2min
                                "oflags" : ["post", "fciq"],
                                "validate" : True}
                
                cash = price*volume*1.015

                if side == "buy":
                    orders[index]["close_ordertype"] = "stop-loss-limit"
                    orders[index]["close_price"] = round(price*0.95, self.price_decimals[currency])
                    orders[index]["close_price2"] = round(price*0.94, self.price_decimals[currency])

                try:
                    order = self.auth_trade.create_order(**orders[index])
                    logging.info(f"[TRADING][VALIDATE] order {order['descr']} validated | {cash:.2f} EUR")
                    orders[index]["validate"] = False

                except Exception as e:
                    logging.error(f"[TRADING] {e}")
                
            else:
                if volume < min_volume:
                    logging.warning(f"[{side} {pair}] WONT BE SENT ORDER Volume :{volume} < {min_volume}")
                if comment_price == "NO_PASS":
                    logging.warning(f"[{side} {pair}] WONT BE SENT ORDER PRICE : {price} vs {tick_price}")

        return orders
    
    def cancel_orders(self):
        nbr_orders = self.auth_trade.cancel_all_orders()
        logging.info(f"[TRADING] cancelled all orders {nbr_orders}")
        if nbr_orders["count"]>0:
            time.sleep(2)

    def pass_orders(self, orders):

        passed_orders = []
        for _, order in orders.items(): 
            if not order["validate"]:
                order = self.auth_trade.create_order(**order)
                
                if "error" in order.keys() > 0:
                    logging.error(f"[TRADING][ORDER] Passed order failed {order}")
                else:
                    logging.info(f"[TRADING][PASSED] order {order['descr']}")
                    passed_orders +=order['txid']
        
        return passed_orders
    

    def get_orders_info(self, list_id_orders):

        orders_status = {}
        if len(list_id_orders):
            orders = self.auth_user.get_orders_info(txid=list_id_orders)

            for k, order in orders.items():
                orders_status[k] = {'refid' : order['refid'],
                                    "status" : order['status'],
                                    "reason" : order["reason"],
                                    "opentm" : datetime.fromtimestamp(order['opentm']),
                                    'pair' : order['descr']['pair'],
                                    'vol_exec' : order['vol_exec'],
                                    'price_order' : order['descr']['price'],
                                    'price_exec' : order['price'],
                                    'cost' : order['cost'],
                                    'fee' : order['fee'],
                                    'limitprice' : order['limitprice'],
                                    'descr' : order['descr']['order'],
                                    'cash' : float(order['vol_exec'])*float(order['price'])}
                
                if "closetm" in order.keys():
                    orders_status[k]["closetm"] = datetime.fromtimestamp(order['closetm'])
                
            orders_status = pd.DataFrame(orders_status).T
            orders_status.columns = smart_column_parser(orders_status.columns)

            return orders_status
        
        return pd.DataFrame()
            