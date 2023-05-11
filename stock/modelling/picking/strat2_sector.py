import pandas as pd 
import re
from modelling.topic_extraction.TopicExtraction import TopicExtraction
import matplotlib.pyplot as plt 
from sklearn.neighbors import NearestNeighbors
from nltk import sent_tokenize
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

def filter_nouns(x):
    tagged_sentence = nltk.tag.pos_tag(x.split())
    edited_sentence = [word for word,tag in tagged_sentence if tag != 'NNPS'] # and tag != 'NNPS']
    return " ".join(edited_sentence)


def handle_sentences(x, i):
    sentences = sent_tokenize(x)[:-1]
    based = sent_tokenize(x)[0].split(" based ")
    if len(based) > 1:
        based = based[1]
    else:
        based = based[0]

    return based + ". ".join(sentences[1:i])


def company_neighbors(df, tp, n_neighbors = 10):

    mapped_name = df["index"].to_dict()
    text_splits = [5, 8, 11]

    for enum, i in enumerate(text_splits):

        df["FILL_DESC"] = df["PROFILE_DESC"].apply(lambda x : handle_sentences(x, i))
        
        df = tp.clean_text(df, target="FILL_DESC")
        df.index = df["ID_TEXT"].values
        tp.update_parameters_small({})
        df["CLEAN_TEXT"] = df[["CLEAN_TEXT", "PROFILE_COMPANY_NAME"]].apply(lambda x: re.sub(x[1], "", x[0]), axis=1)
        
        # if i > 1:
        #     vectorizer = TfidfVectorizer(ngram_range=(1,1))
        #     X = vectorizer.fit_transform(df["CLEAN_TEXT"])
        #     idf = vectorizer.idf_
        #     words_importance= pd.DataFrame([vectorizer.get_feature_names(), idf]).T.sort_values(1, ascending=0)

        #     # remove most occuring words and least occuring words
        #     threshold_too_many = np.percentile(words_importance[1], 0.02)
        #     threshold_too_few = np.percentile(words_importance[1], 99.5)
        #     to_remove = words_importance.loc[(words_importance[1] <= threshold_too_many)|(words_importance[1] >= threshold_too_few), 0].tolist()  

        #     filtered_text = df["CLEAN_TEXT"].apply(lambda x : " ".join([word for word in x.split() if word not in to_remove]))
        # else:
        filtered_text = df["CLEAN_TEXT"]

        embeddings =  tp.embedding_reduction(filtered_text, umap=False)
        proximity_score = tp.calculate_similarity(embeddings, embeddings)

        proximity_score = pd.DataFrame(proximity_score, columns = range(df.shape[0]), index=range(df.shape[0]))
        proximity_score.columns = pd.Series(proximity_score.columns).map(mapped_name)
        proximity_score.index = pd.Series(proximity_score.index).map(mapped_name)   

        if enum > 0:
            full_proximity = proximity_score + full_proximity
        else:
            full_proximity = proximity_score

        # proximity_score.nlargest(10, 'cnp assurances sa')['cnp assurances sa']
    
    full_proximity = full_proximity / len(text_splits)

    # construct final dataset 
    neighbors = pd.DataFrame(index= full_proximity.index, columns= ["NEIGHBORS", "WEIGHTS"])
    for company in neighbors.index:
        neighbors.loc[company, "NEIGHBORS"] = list(full_proximity.nlargest(n_neighbors + 1, company)[company].index[1:])
        neighbors.loc[company, "WEIGHTS"] = list(full_proximity.nlargest(n_neighbors + 1, company)[company].values[1:])

    return neighbors


def closest_per_sector(final_inputs, data, n_neighbors = 15) -> pd.DataFrame:
    """
    TODO: check those with low weights for all neighbors -> change them 

    Args:
        final_inputs ([pd.DataFrame]): [description]
        data ([pd.DataFrame]): [description of each stock]
        n_neighbors (int, optional): [description]. Defaults to 15.

    Returns:
        [pd.DataFrame]: [frame with neighbors]
    """

    df = final_inputs["results"][["PROFILE_DESC", "SPECIFIC", "PROFILE_COMPANY_NAME"]].copy()
    df = df.loc[~df["PROFILE_DESC"].isnull()]
    df = df.loc[~df.index.isnull()]
    df = df.reset_index()
    n_neighbors = 15

    # merge to have the sector
    df = pd.merge(df, data, left_on="index", right_on="REUTERS_CODE", how="left", validate="1:1")
    
    remove_words = ["france", "german", "germany", "austrian", "english", "italian", "italy",
                    "french", "sweden", "austria", "belgium", "spain", "spanish",
                    "luxembourg", "australian", "australia", "china", "london", "united", "kingdom",
                    "norway", "based", "company", "switzerland", "japan", "netherlands",
                    "international", "korea", "finland"]

    tp = TopicExtraction(remove_words=remove_words)

    all_results = pd.DataFrame()

    for sector in df["SECTOR"].unique():
        sub_sector = df.loc[df["SECTOR"] == sector]
        sub_sector = sub_sector.reset_index(drop=True)
        all_results = pd.concat([all_results, company_neighbors(sub_sector, tp, n_neighbors = n_neighbors)], axis=0)

    return all_results
        