import pandas as pd 
import re
from modelling.topic_extraction.TopicExtraction import TopicExtraction
import matplotlib.pyplot as plt 
from sklearn.neighbors import NearestNeighbors
from nltk import sent_tokenize
import nltk

def filter_nouns(x):
    tagged_sentence = nltk.tag.pos_tag(x.split())
    edited_sentence = [word for word,tag in tagged_sentence if tag != 'NNP'] # and tag != 'NNPS']
    return " ".join(edited_sentence)


def company_neighbors(final_inputs, n_neighbors = 11):
    # TODO: Weight should depend on Total revenue of both attracted company

    df = final_inputs["results"][["DESC", "SPECIFIC"]].copy()
    df = df.loc[~df["DESC"].isnull()]

    # df["COMPANY_NAME"] = df["DESC"].apply(lambda x : re.sub(r'\([^)]*\)', ' ', str(x).split(" is ")[0]).split(",")[0])
    # df["FILL_DESC"] = df["DESC"].apply(lambda x : str(x).split(" is ")[1])
    df["FILL_DESC"] = df["DESC"].apply(lambda x : ". ".join(sent_tokenize(x)[:-1]))
    df["FILL_DESC"] = df["FILL_DESC"].apply(lambda x : filter_nouns(x))

    tp = TopicExtraction()
    df = tp.clean_text(df, target="FILL_DESC")
    df.index = df["ID_TEXT"].values
    tp.update_parameters_small({})
    embeddings =  tp.embedding_reduction(df["CLEAN_TEXT"], umap=False)

    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(embeddings)
    mapping_words = neigh.kneighbors(embeddings)

    distance = pd.DataFrame(mapping_words[0])
    plt.plot(distance.columns, distance.mean(axis=0))

    for i in range(n_neighbors): 
        df[f"NEIGHBOR_{i}"] = 0
        df[f"WEIGHT_NEIGH_{i}"] = 0

    df[[x for x in df.columns if "NEIGHBOR_" in x]] = mapping_words[1]
    df[[x for x in df.columns if "WEIGHT_NEIGH_" in x]] = 1/mapping_words[0]**2
    mapping_id_name = df.reset_index()["index"].to_dict()

    for col in [x for x in df.columns if "NEIGHBOR_" in x]:
        df[col] = df[col].map(mapping_id_name)

    del df["NEIGHBOR_0"]
    del df["WEIGHT_NEIGH_0"]

    return df
