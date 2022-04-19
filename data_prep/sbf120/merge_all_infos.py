import pandas as pd 

def final_data_prep(data, final_inputs):

    res =  final_inputs["results"]
    neigh =  final_inputs["neighbors"]

    final = pd.merge(res, neigh, left_index=True, right_index=True, how="left", validate="1:1")  

    data = data[["REUTERS_CODE", "SECTOR", "SUB INDUSTRY", "NAME", "Country"]]
    final = pd.merge(final, data, left_index=True, right_on="REUTERS_CODE", how="left", validate="1:1")  

    final.index = list(final["REUTERS_CODE"])
    
    return final
