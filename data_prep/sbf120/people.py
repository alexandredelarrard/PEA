import numpy as np 


def people_analysis(inputs, analysis_dict):
    """TODO: 
    - Men vs Women in board
    - parcours du ceo (# annees passee dans la boite avant d'arriver)
    - si ceo = fondateur ?

    Args:
        inputs ([type]): [description]
        analysis_dict ([type]): [description]

    Returns:
        [type]: [description]
    """

    data = inputs["PEOPLE"].copy()
    data = data.loc[~data["NAME"].isnull()]

    new_results = {}

    if data.shape[0] > 0:
        # get number independent directors
        new_results["TEAM_#_INDEPENDENT_DIRECTOR"] = data.loc[data["POSITION"] == "Independent Director"].shape[0]
        data =  data.loc[data["POSITION"] != "Independent Director"]

        if data.shape[0] > 0:
            # replace age missing 
            for col in ["AGE", "APPOINTED"]:
                data[col] = data[col].apply(lambda x : str(x).replace("--", "0")).astype(int)
                data[col] = np.where(data[col] == 0, np.nan, data[col])

            is_ceo = data["POSITION"].apply(lambda x: "Chief Executive Officer," in x)
            is_cfo = data["POSITION"].apply(lambda x: "Chief Financial Officer," in x)
            is_coo = data["POSITION"].apply(lambda x: "Chief Operating Officer," in x)

            new_results["TEAM_CEO_NAME"] = data.loc[0, "NAME"]
            new_results["TEAM_CEO_APPOINTED"] = data.loc[0, "APPOINTED"]
            new_results["TEAM_C_LEVEL_AVG_APPOINTED"] = data.loc[is_cfo + is_coo + is_ceo, "APPOINTED"].mean()
            new_results["TEAM_LEADER_AGE_AVG"] = data["AGE"].mean()
            new_results["TEAM_LEADER_APPOINTED_AVG"] = data["APPOINTED"].mean()

    analysis_dict.update(new_results)

    return analysis_dict
