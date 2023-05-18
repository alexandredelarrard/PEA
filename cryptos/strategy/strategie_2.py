
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from strategy.main_strategie import MainStrategy


class Strategy2(MainStrategy):

    def __init__(self, configs, start_date, end_date =None):

        MainStrategy.__init__(self, configs, start_date, end_date)

        self.picked_strategie = self.main_strategy_2

        # TODO / TEST
        # - probabilité de hausse next 0.25, 0.5, 1, 2, 3, 4, 5, 10, 15 jours (modelling) -> not BTC / ETH becasue is market 
        #   - diff to mean 0.25, 0.5, etc. 
        #   - std 0.25, etc. 
        #   - diff to market 
        #   - diff to s&p500 
        #   - diff to gold 
        #   - news feeds -> sentiments strengths 
        #   - twitter feeds 
        #   - etoro feeds ? 
        #   - spread bid / ask + tradecount std / min / max 
        #   - % of all coins traded over past X times 
        #   - remaining liquidity / coins to create (offer)
        #   - Coin tenure / descriptors (clustering ?) / employees ? / state, etc.
        # - RSI
        # - spread bid ask over time 
        # - tradecount over time
        # - news sentiments extract 
        # - coin trends (volume tradecount var + news attraction) -> for new coins only 
        # - put / call futurs baught same time -> price on vol 
        # - diff to market index -> buy the lower one, sell the higher one 

    def main_strategy_2(self, prepared, currency = "BTC",
                        df_init=None, lag=None, variable="TARGET"):
        
        return pd.DataFrame(), pd.DataFrame()
