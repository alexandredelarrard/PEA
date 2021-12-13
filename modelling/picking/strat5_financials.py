import pandas as pd 
import numpy as np

def scoring_financials(final_inputs):

    results = final_inputs["results"]
    scoring = results[["RATING"]]

    # scoring analysts ratings 
    results["RATING"].fillna(results["RATING"].median(), inplace=True)
    results["RATING"] = np.where(results["RATING_NBR_ANALYSTS"] <= 5, results["RATING"].median(), results["RATING"])
    results["RATING"] = 6 - results["RATING"]
    scoring["RATING"], scoring["INDUSTRY_RATING"] = deduce_ranking(final_inputs, "RATING")

    # reuters forward PE 

    # scoring growth 
    scoring["3Y_GROWTH"], scoring["INDUSTRY_3Y_GROWTH"] = deduce_ranking(final_inputs, "3Y_AVG_GROWTH_CA_ANNUAL")

    # scoring profits 
    scoring["3Y_GROWTH"], scoring["INDUSTRY_3Y_GROWTH"] = deduce_ranking(final_inputs, "3Y_AVG_GROWTH_CA_ANNUAL")


    # EV_EBITDA


    return scoring
    

def deduce_ranking(final_inputs, var):

    df_nei =  final_inputs["neighbors"]

    neigh_cols = [x for x in df_nei.columns if "NEIGHBOR_" in x]
    neighbors = df_nei[neigh_cols]
    neighbors["INDEX"] = neighbors.index

    weight_cols = [x for x in df_nei.columns if "WEIGHT_NEIGH_" in x]
    weight_neighbors = df_nei[weight_cols]
    cst = weight_neighbors.sum(axis=1)
    for col in weight_cols:
        weight_neighbors[col] = weight_neighbors[col] / cst

    mapping = final_inputs["results"][var].to_dict()
    for col in neighbors.columns:
        neighbors[col] = neighbors[col].map(mapping)

    weight_neighbors.columns = neigh_cols
    mean_var = (weight_neighbors*neighbors[neigh_cols]).sum(axis=1)

    distance_to_mean = (neighbors["INDEX"] - mean_var)*100/mean_var
    
    return distance_to_mean, mean_var