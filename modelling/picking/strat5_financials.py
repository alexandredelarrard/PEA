import pandas as pd 
import numpy as np


def business_rules(scoring):

    points = pd.DataFrame(index= scoring.index)

    # points top line 
    points['LONG_TREND_TOTAL_REVENUE_0'] = np.where(scoring['LONG_TREND_TOTAL_REVENUE_0'] > 50, 4, 
                                        np.where(scoring['LONG_TREND_TOTAL_REVENUE_0'] > 30, 3, 
                                        np.where(scoring['LONG_TREND_TOTAL_REVENUE_0'] > 10, 2, 
                                        np.where(scoring['LONG_TREND_TOTAL_REVENUE_0'] < 0, -2, 0))))

    points['LONG_TREND_NET_INCOME_BEFORE_EXTRA_ITEMS_0'] = np.where(scoring['LONG_TREND_NET_INCOME_BEFORE_EXTRA_ITEMS_0'] > 50, 4, 
                                            np.where(scoring['LONG_TREND_NET_INCOME_BEFORE_EXTRA_ITEMS_0'] > 30, 3, 
                                            np.where(scoring['LONG_TREND_NET_INCOME_BEFORE_EXTRA_ITEMS_0'] > 10, 2, 
                                            np.where(scoring['LONG_TREND_NET_INCOME_BEFORE_EXTRA_ITEMS_0'] < 0, -2, 0))))

    points['%_YOY_TOTAL_REVENUE_0'] = np.where(scoring['%_YOY_TOTAL_REVENUE_0'] > 50, 3, 
                                         np.where(scoring['%_YOY_TOTAL_REVENUE_0'] > 30, 2, 
                                         np.where(scoring['%_YOY_TOTAL_REVENUE_0'] > 10, 1, 0)))

    points['TREND_OP_INCOME / TREND_REVENUE'] = np.where(scoring['TREND_OP_INCOME / TREND_TOTAL_REVENUE_0'] > 20, 2, 
                                         np.where(scoring['TREND_OP_INCOME / TREND_TOTAL_REVENUE_0'] > 10, 1, 0))

    points['%_YOY_NET_INCOME_BEFORE_EXTRA_ITEMS_0'] = np.where(scoring['%_YOY_NET_INCOME_BEFORE_EXTRA_ITEMS_0'] > 50, 2, 
                                                        np.where(scoring['%_YOY_NET_INCOME_BEFORE_EXTRA_ITEMS_0'] > 30, 1, 
                                                        np.where(scoring['%_YOY_NET_INCOME_BEFORE_EXTRA_ITEMS_0'] > 10, 0.5, 0)))

    points['%_YOY_R&D_0'] = np.where(scoring['%_YOY_R&D_0'] > 30, 3, 
                        np.where(scoring['%_YOY_R&D_0'] > 20, 2, 
                        np.where(scoring['%_YOY_R&D_0'] > 10, 1, 0)))

    # points pe / intrinsic value 
    points['P/E_PREDICTION'] = np.where(scoring['P/E_PREDICTION']*(1+scoring['%_ERROR_SECTOR']/100) - scoring['_P/E_+1Q_Y-0'] > 10, 6, 
                                         np.where(scoring['P/E_PREDICTION']*(1+scoring['%_ERROR_SECTOR']/100) - scoring['_P/E_+1Q_Y-0'] > 5, 4, 
                                         np.where(scoring['P/E_PREDICTION']*(1+scoring['%_ERROR_SECTOR']/100) - scoring['_P/E_+1Q_Y-0'] > 2, 1, 0)))

    points['PROFILE_FORWARD_P_E'] = np.where(scoring['PROFILE_FORWARD_P_E'] - scoring['_P/E_+1Q_Y-0'] <= -10, 6, 
                                    np.where(scoring['PROFILE_FORWARD_P_E'] - scoring['_P/E_+1Q_Y-0'] <= -5, 4, 
                                    np.where(scoring['PROFILE_FORWARD_P_E'] - scoring['_P/E_+1Q_Y-0'] < -3, 1, 0)))

    points['DISTANCE MARKET_CAP / INTRINSIC'] = np.where(scoring['DISTANCE MARKET_CAP / INTRINSIC _0'] < 30, 8, 
                                                    np.where(scoring['DISTANCE MARKET_CAP / INTRINSIC _0'] < 50, 4, 
                                                    np.where(scoring['DISTANCE MARKET_CAP / INTRINSIC _0'] < 75, 3, 
                                                    np.where(scoring['DISTANCE MARKET_CAP / INTRINSIC _0'] < 100, 2, 
                                                    np.where(scoring['DISTANCE MARKET_CAP / INTRINSIC _0'] < 200, 1, 
                                                    np.where(scoring['DISTANCE MARKET_CAP / INTRINSIC _0'] > 500, -2, 0))))))

    points["PROFILE_RATING"] = np.where(scoring["PROFILE_RATING"] > 3, 2, 
                               np.where(scoring["PROFILE_RATING"] > 2.5, 1, 0))

    # STOCK
    points['STOCK_%_52_WEEKS_0'] = np.where(scoring['STOCK_%_52_WEEKS_0'] < 0.7*scoring['%_YOY_NET_INCOME_BEFORE_EXTRA_ITEMS_0'], 3, 
                                    np.where(scoring["STOCK_%_52_WEEKS_0"] < 0.9*scoring['%_YOY_NET_INCOME_BEFORE_EXTRA_ITEMS_0'], 2, 
                                    np.where(scoring["STOCK_%_52_WEEKS_0"] < scoring['%_YOY_NET_INCOME_BEFORE_EXTRA_ITEMS_0'], 1, 0)))

    # balance sheet & cash 
    points['BALANCE_%_SHARE_GOODWILL_ASSETS_0'] = np.where(scoring['BALANCE_%_SHARE_GOODWILL_ASSETS_0'] > 50, 2, 
                            np.where(scoring['BALANCE_%_SHARE_GOODWILL_ASSETS_0'] > 30, 1.5, 
                            np.where(scoring['BALANCE_%_SHARE_GOODWILL_ASSETS_0'] > 10, 1, 0)))
    
    points['CASH_%_INTO_ACQUISITION_0'] = np.where(scoring['CASH_%_INTO_ACQUISITION_0'] > 50, 2, 
                            np.where(scoring['CASH_%_INTO_ACQUISITION_0'] > 30, 1.5, 
                            np.where(scoring['CASH_%_INTO_ACQUISITION_0'] > 10, 1, 0)))

    points['CASH_%_INTO_ACTIVITY_0'] = np.where(scoring['CASH_%_INTO_ACTIVITY_0'] > 50, 2, 
                            np.where(scoring['CASH_%_INTO_ACTIVITY_0'] > 30, 1.5,  
                            np.where(scoring['CASH_%_INTO_ACTIVITY_0'] > 10, 1, 0)))

    points['BALANCE_TREND_SHAREHOLDERS_EQUITY_0'] = np.where(scoring['BALANCE_TREND_SHAREHOLDERS_EQUITY_0'] > 50, 2, 
                                            np.where(scoring['BALANCE_TREND_SHAREHOLDERS_EQUITY_0'] > 30, 1.5, 
                                            np.where(scoring['BALANCE_TREND_SHAREHOLDERS_EQUITY_0'] > 10, 1, 
                                            np.where(scoring['BALANCE_TREND_SHAREHOLDERS_EQUITY_0'] < 0, -1, 
                                            0))))

    points['BALANCE_%_CURRENT_ASSET / CURRENT_DEBT_0'] = np.where(scoring['BALANCE_%_CURRENT_ASSET / CURRENT_DEBT_0'] > 250, 2, 
                                                        np.where(scoring['BALANCE_%_CURRENT_ASSET / CURRENT_DEBT_0'] > 200, 1.5, 
                                                        np.where(scoring['BALANCE_%_CURRENT_ASSET / CURRENT_DEBT_0'] > 150, 1, 0)))

    # competitors  
    points['%_DISTANCE__P/E_Y-0'] =np.where(scoring['%_DISTANCE__P/E_Y-0'] < -0.3, 2, 
                                    np.where(scoring['%_DISTANCE__P/E_Y-0'] < 0, 1, 0))
    
    points['%_DISTANCE_%_NET_PROFIT_MARGIN_0'] = np.where(scoring['%_DISTANCE_%_NET_PROFIT_MARGIN_0'] > 1, 2, 
                                                np.where(scoring['%_DISTANCE_%_NET_PROFIT_MARGIN_0'] > 0.5, 1, 0))

    points['%_DISTANCE_%_YOY_TOTAL_REVENUE_0'] = np.where(scoring['%_DISTANCE_%_YOY_TOTAL_REVENUE_0'] > 1, 2, 
                                                np.where(scoring['%_DISTANCE_%_YOY_TOTAL_REVENUE_0'] > 0.5, 1, 0))

    final = pd.DataFrame(points.sum(axis=1).sort_values(ascending=False))
    final.columns = ["BUSINESS_RULE"]

    scoring = pd.merge(scoring, final, left_index=True, right_index=True, how="left", validate="1:1")
  
    return scoring