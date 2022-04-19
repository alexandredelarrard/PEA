# import logging
import warnings
import streamlit as st
import UI.web_app as ui
import UI.SessionState as SessionState

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")
col1, col2, col3 = st.columns([2,2,2])


def hash_data(TopicExtraction):
    industry = TopicExtraction.industry
    dictionnary = TopicExtraction.dictionnary
    shape = TopicExtraction.data.shape[0]
    params = TopicExtraction.params
    remove_words = TopicExtraction.remove_words
    target= TopicExtraction.target
    return (industry, dictionnary, shape, params, remove_words, target)


def main(params):

    col2.title("Stock Picking")

    session_state = SessionState.get(filename="")
    error_comments, results, widget_ui = ui.get_inputs(st, session_state)

    # display results
    ui.display_results(st, results, widget_ui)

    # # download button 
    # to_save = top_cluster.save_all(top_cluster.data, top_cluster.df_top_n_words, top_cluster.to_fix)
    # st.markdown(ui.get_table_download_link(data["OUTPUT"], to_save), unsafe_allow_html=True)

if __name__ == "__main__":
    params = {"n_neighbors" : 10}
    main(params)