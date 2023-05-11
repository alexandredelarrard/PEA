import os 
import pandas as pd 
import streamlit as st
from utils.general_functions import smart_column_parser


def load_data(path):

    error = ""
    df = pd.DataFrame()
    filename, file_extension = os.path.splitext(path.name)
    
    if file_extension == ".csv":
        df = pd.read_csv(path, sep=None)
        df.columns = smart_column_parser(df.columns)

    elif file_extension == ".xlsx":
        df = pd.read_excel(path)
        df.columns = smart_column_parser(df.columns)

    else:
        error = f"{filename} must be a .csv or .xlsx with a unique sheet"

    return df, error


def check_dictionnary(dico_ps):

    error = ""
    if dico_ps.shape[1]>2:

        #clean and return cluster dataframe
        results = pd.DataFrame()
        for col in dico_ps.columns:
            results = results.append([[col, dico_ps[col].tolist()]])

        results.columns = ["CATEGORY", "WORDS"]
        results = results.explode("WORDS")
        results = results.loc[~results["WORDS"].isnull()].reset_index(drop=True)
    
    else:
        error = "Please provide a seed word dictionnary with at least 3 clusters"
        results = pd.DataFrame([])

    return results, error


def create_warning(st, data, errors):

    for error in errors:
        if len(error)>0:
            st.write(error)

    if "COMMENTS" not in data.keys():
        st.error('Please provide comments to cluster !')

    if "OUTPUT" == "":
        st.error('Please provide variable name to analyse ')
    
    Warnings = ""
    if data["INDUSTRY"].shape[0] == 0:
        Warnings += "No industry words corrections have been provided ! \n"
    
    if data["DICTIONARY"].shape[0] == 0:
            Warnings += "No pre defined clusters provided ! \n"

    if len(data["remove_stop_words"]) == 0 :
            Warnings += "No additional stop words provided ! \n"

    return Warnings


def fill_data(widgets):
    
    _, dataframe, selected_filename, uploaded_file_industry, uploaded_file_category, user_input = widgets
    error_dict, error_dico, error_ind = ["", "", ""]

    data = {"remove_stop_words" : [],
            "INDUSTRY" : pd.DataFrame([]),
            "DICTIONARY": pd.DataFrame([]),
            "OUTPUT" : selected_filename,
            "COMMENTS" : dataframe}

    if uploaded_file_category is not None:
        data["DICTIONARY"], error_dict = load_data(uploaded_file_category)

    if data["DICTIONARY"].shape[0]>0:
        data["DICTIONARY"], error_dico = check_dictionnary(data["DICTIONARY"])

    if uploaded_file_industry is not None:
        data["INDUSTRY"], error_ind= load_data(uploaded_file_industry)

    data["remove_stop_words"] = [x.strip() for x in user_input.lower().split(",")]

    return data, [error_dict, error_dico, error_ind]
