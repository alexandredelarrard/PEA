import pandas as pd 
import streamlit as st
import base64
import numpy as np

def get_inputs(st, session_state):

    error_comments = ""
    the_country = "US"
    
    st.sidebar.header("Company to check")

    results = pd.read_csv(r"C:\Users\de larrard alexandre\Documents\repos_github\PEA\data\results\financials_V2.csv", sep=";")
    results.index = list(results["Unnamed: 0"])
    del results["Unnamed: 0"]
    scores = results["BUSINESS_RULE"].astype(float)
    results.columns = [x.upper() for x in results.columns]

    # add slider 
    business_rank = st.sidebar.slider('What score are you looking at ?', int(scores.min()-1), int(scores.max()+1), 15)

    # add region box selection 
    st.sidebar.header('SELECT STOCK REGIONS')
    specific_country = st.sidebar.checkbox('SELECT COUNTRY', value =False)

    # add region box selection 
    st.sidebar.header('SELECT SECTOR')
    specific_sector = st.sidebar.checkbox('SELECT SECTOR', value =False)

    constraints = results["BUSINESS_RULE"] >= business_rank

    if specific_country:
        the_country = st.sidebar.selectbox('Select country', np.sort(results["COUNTRY"].unique()))
        constraints = constraints&(results["COUNTRY"] == the_country)

    if specific_sector:
        the_sector = st.sidebar.selectbox('Select sector', np.sort(results["SECTOR"].unique()))
        constraints = constraints&(results["SECTOR"] == the_sector)

    columns = results.loc[constraints, "NAME"].tolist()

    columns = np.sort(columns)
    # add company box
    selected_filename = st.sidebar.selectbox('Select company', columns)

    for col in results.columns:
        try: 
            results[col] = results[col].astype(float).round(4)
        except Exception:
            pass 

    return error_comments, results, [selected_filename]


def display_results(st, results, widget_ui):
    
    company_name = widget_ui[-1]
    company = results.loc[results["NAME"] == company_name].index[0]

    neighbor = eval(results.loc[company, "NEIGHBORS"])[:9]
    sub_df = results.loc[[company] + neighbor]
    sub_df.index = sub_df["NAME"].values

    st.header(f"{company_name.upper()} / {results.loc[company, 'SECTOR']}")

    st.header("COMPETITORS")
    pe = ['SUB INDUSTRY', 
        'REUTERS_CODE', 
        "LATEST_DATE_FINANCIALS",
        "PROFILE_DESC", 'TEAM_CEO_APPOINTED', 
        'TEAM_LEADER_AGE_AVG', 'TEAM_LEADER_APPOINTED_AVG',
        "BUSINESS_RULE",
        'DISTANCE MARKET_CAP / INTRINSIC _0',
        'PROFILE_FORWARD_P_E', 
        "P/E_PREDICTION",
        '_P/E_Y-0', 
        '_P/E_Y-1',
        '_P/E_Y-2', 
        '_P/E_Y-3',
        "INFLATION",
        'PROFILE_RATING',
        'PROFILE_RATING_NBR_ANALYSTS',
        'STOCK_%_52_WEEKS_0',
       'STOCK_% TREND_NET_INCOME_WO_EXTRA - STOCK_TREND_0',
       '%_YOY_TOTAL_REVENUE_0', 
       '%_YOY_OPERATING_INCOME_0',
       '%_NET_PROFIT_MARGIN_0',
       '%_R&D_IN_OPERATING_INCOME_0'
       ]

    st.dataframe(sub_df[pe].astype(str).T, 1600, 1500)

    add_graph_top_bottom_line(results, company)
    add_graph_bilan(results, company)

    st.header("P/E")
    pe = ['PROFILE_FORWARD_P_E', "P/E_PREDICTION", "_P/E_+1Q_Y-0",
        '_P/E_Y-0', '_P/E_Y-1', '_P/E_Y-2', '_P/E_Y-3']
    st.line_chart(sub_df[pe].T.clip(0, 85), 1500, 750)

    col1, col2, col3= st.columns([2,2, 2])

    col1.header("REVENUE/INCOME")
    col = "%_YOY_TOTAL_REVENUE"
    pe = []
    for i in range(4):
        pe.append(f'{col}_{i}')
    col1.line_chart(sub_df[pe].T, 1200, 750)

    col2.header("INCOME")
    col = '%_YOY_OPERATING_INCOME'
    pe = []
    for i in range(4):
        pe.append(f'{col}_{i}')
    col2.line_chart(sub_df[pe].T, 1200, 750)

    col3.header("% MARGIN")
    col = '%_NET_PROFIT_MARGIN'
    pe = []
    for i in range(4):
        pe.append(f'{col}_{i}')
    col3.line_chart(sub_df[pe].T, 1200, 750)
        


def add_graph_top_bottom_line(results, company):

    col1, col2, col3 = st.columns([2,2, 2])
    for_graph = pd.DataFrame(index=range(4)[::-1])

    for col in ["_TOTAL_REVENUE", '_OPERATING_INCOME', '_NET_INCOME_BEFORE_EXTRA_ITEMS', '_R&D']:
        for_graph[col] = results.loc[company][[f'{col}_0',
                                                f'{col}_1',
                                                f'{col}_2',
                                                f'{col}_3']].values

    delta_graph = pd.DataFrame(index=range(4)[::-1])
    for col in ['%_YOY_TOTAL_REVENUE', '%_YOY_OPERATING_INCOME', '%_YOY_NET_INCOME_BEFORE_EXTRA_ITEMS', '%_YOY_R&D']:
        delta_graph[col] = results.loc[company][[f'{col}_0',
                                                f'{col}_1',
                                                f'{col}_2',
                                                f'{col}_3']].values

    percent_graph = pd.DataFrame(index=range(4)[::-1])
    for col in ['%_NET_PROFIT_MARGIN', '%_R&D_IN_OPERATING_INCOME', '%_SALES_GENERAL_IN_REVENUE', '%_TAXES_&_EXTRA_IN_TOTAL_REVENUE']:
        percent_graph[col] = results.loc[company][[f'{col}_0',
                                                f'{col}_1',
                                                f'{col}_2',
                                                f'{col}_3']].values
    
    col1.header("TOP line")
    col1.line_chart(for_graph, height=500, width=600)

    col2.header("DELTA TOP line in %")
    col2.line_chart(delta_graph, height=500, width=600)

    col3.header("% of Revenue")
    col3.line_chart(percent_graph, height=500, width=600)



def add_graph_bilan(results, company):

    col1, col2, col3 = st.columns([2,2, 2])
    for_graph = pd.DataFrame(index=range(4)[::-1])

    for col in ['BALANCE_%_CASH / TOTAL_DEBT',
                'BALANCE_%_TOTAL_ASSET / TOTAL_LIABILITIES', 
                'BALANCE_%_CURRENT_ASSET / CURRENT_DEBT', 
                'BALANCE_%_SHARE_GOODWILL_ASSETS']:
        for_graph[col] = results.loc[company][[f'{col}_0',
                                                f'{col}_1',
                                                f'{col}_2',
                                                f'{col}_3']].values

    delta_graph = pd.DataFrame(index=range(4)[::-1])
    for col in ['CASH_%_DIVIDENDS IN ACTIVITY_CASH', 'CASH_%_FINANCE IN ACTIVITY_CASH',
                 'CASH_%_INTO_ACQUISITION', 'CASH_%_INTO_ACTIVITY']:
        delta_graph[col] = results.loc[company][[f'{col}_0',
                                                f'{col}_1',
                                                f'{col}_2',
                                                f'{col}_3']].values

    percent_graph = pd.DataFrame(index=range(4)[::-1])
    for col in ['BALANCE_TREND_CURRENT_ASSET / CURRENT_DEBT', 'BALANCE_TREND_SHAREHOLDERS_EQUITY', 
                'CASH_TREND_FREE_CASH_FLOW']:
        percent_graph[col] = results.loc[company][[f'{col}_0',
                                                f'{col}_1',
                                                f'{col}_2',
                                                f'{col}_3']].values
    
    col1.header("BALANCE")
    col1.line_chart(for_graph, height=500, width=600)

    col2.header("CASH")
    col2.line_chart(delta_graph, height=500, width=600)

    col3.header("TRENDS")
    col3.line_chart(percent_graph, height=500, width=600)


def get_table_download_link(output, val):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    b64 = base64.b64encode(val)  
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{output}.xlsx">Download {output}</a>'
