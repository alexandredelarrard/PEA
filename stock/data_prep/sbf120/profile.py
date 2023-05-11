
from utils.general_cleaning import create_index, load_currency


def profile_analysis(inputs, analysis_dict, params):

    data = inputs['PROFILE'].copy()
    data = create_index(data)
    df_currency =  load_currency(params, "currency_stock")
    currency = ""

    new_results = {}
    if "COMPANY_NAME" in data.index:
        new_results["PROFILE_COMPANY_NAME"] = data.loc["COMPANY_NAME"][0].lower().replace("about ","")
    else:
        new_results["PROFILE_COMPANY_NAME"] = data.loc["PRESENTATION"][0].lower().split(" is a")[0]

    new_results["PROFILE_MARKET_CAP"] = float(data.loc["MARKET_CAP_MIL"][0].replace(",",""))
    new_results["PROFILE_DESC"] = data.loc["PRESENTATION"][0]
    
    if "SHARES_OUT_MIL" in data.index:
        new_results["PROFILE_SHARES_OUT_MIL"] = float(data.loc["SHARES_OUT_MIL"][0].replace(",",""))

    if "FORWARD_P_E" in data.index:
        if data.loc["FORWARD_P_E"][0] != "--":
            new_results["PROFILE_FORWARD_P_E"] = float(data.loc["FORWARD_P_E"][0].replace(",",""))

    if "RATING" in data.index:
        new_results["PROFILE_RATING"] = float(data.loc["RATING"][0].split("-")[0].replace(" mean rating ", ""))
        new_results["PROFILE_RATING_NBR_ANALYSTS"] = float(data.loc["RATING"][0].split("-")[1].replace(" analysts", "").strip())

    if "CURRENCY" in data.index:
        extract = data.loc["CURRENCY"][0]
        extract = extract.split(".")[0].split(",")[1].strip()
        if extract in ["GBP", "USD", "EUR", "CHF", "SEK", "DKK", 
                        "PLN", "NOK", "KRW", "TWD", "HKD", "AUD",
                        "CNY", "JPY"]:
            currency = extract

    new_results["PROFILE_MARKET_CAP"] = new_results["PROFILE_MARKET_CAP"]*df_currency.iloc[-1,1]

    analysis_dict.update(new_results)

    return analysis_dict, currency