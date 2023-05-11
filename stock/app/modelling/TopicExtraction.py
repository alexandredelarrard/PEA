import numpy as np
# import logging
import pandas as pd
from modelling.Analysis import AnalysisTopicExtraction
from modelling.CleanText import TextCleaner
from modelling.ClusterTopics import TopicClustering
from modelling.DeepClassifier import classification_fine_tune

class TopicExtraction(TopicClustering, TextCleaner, AnalysisTopicExtraction):

    def __init__(self, st="", remove_words=[], params={}, dictionnary=pd.DataFrame([]), industry=pd.DataFrame([]), target=""):
            
        self.st = st
        if st != "":
            self.bar = st.progress(0)
            self.latest_iteration = st.empty()

        self.update_parameters(params)
        self.dictionnary = dictionnary

        # init super class
        TextCleaner.__init__(self, industry=industry, remove_words=remove_words)
        TopicClustering.__init__(self, params=self.params)
        AnalysisTopicExtraction.__init__(self, target=target)


    def progress(self, text, time):

        if self.st != "":
            self.latest_iteration.text(text)
            self.bar.progress(time)

        
    def update_parameters(self, params):
        self.params = {"ngram_range" : (2,4),
                      "random_seed" : 42,
                      "umap_n_neighbors" : 60,  
                      "umap_n_components" : 60,
                      "min_cluster_size" : 10,
                      "n_words_cluster" : 12,
                      "verbose" : 1}
        self.params.update(params)


    def fit(self, text):

        # clean text first 
        self.data = self.clean_text(data=text, target=self.target, function_progress=self.progress)
        self.progress(f"Text cleaned with {self.remove_words}", 20)
        
        if self.dictionnary.shape[0] > 0:
            self.data = self.append_data_dico(self.data, self.dictionnary)

        # embedding 
        embeddings = self.embedding_reduction(self.data["CLEAN_TEXT"].tolist(), umap=True)
        self.progress("Text embedded", 40)

        # clustering in 2 steps 
        for cluster_method in ["HDBSCAN_CLUSTER", "CAH_CLUSTER"]:
            if cluster_method == "HDBSCAN_CLUSTER":
                # cluster based on distance to neighbors 
                cluster = self.hdbscan_clustering(embeddings)
                self.progress("Text cluster 1/2", 50)

            if cluster_method == "CAH_CLUSTER":
                # improve top words and group iteratively close clusters
                cluster = self.CAH_clustering(self.data, corpus_text_cluster)
                self.progress("Text cluster 2/2", 70)

            if self.params["verbose"] > 1:
                self.visualize_cluster(embeddings, cluster)

            # concatenate all text per same cluster
            self.data, corpus_text_cluster = self.textual_clusters(self.data, cluster, cluster_method)

            # get top n words per cluster based on ngrams 
            corpus_text_cluster, df_top_n_words = self.extract_top_n_words_per_topic(corpus_text_cluster, cluster_method)

        # classification for making robust clustering
        clusters = self.create_output(cluster_method = "CAH_CLUSTER")
        self.progress("Results saved", 100)

        return clusters
        

    def create_output(self, cluster_method):

        # extra classification step to make sure clusters are nice 
        self.data, self.classif_models = classification_fine_tune(self.data, self.bert_embeddings, cluster_method, self.progress)      
        self.progress("Text calssified", 90)

        # concatenate all text per same cluster
        self.data, corpus_text_cluster = self.textual_clusters(self.data, self.data["FINAL_CLUSTER"], "FINAL_CLUSTER")

        # get top n words per cluster based on ngrams 
        self.corpus_text_cluster, self.df_top_n_words = self.extract_top_n_words_per_topic(corpus_text_cluster, "FINAL_CLUSTER")

        # save and format results
        output = self.data.loc[self.data["INDEX"] != -1][["INDEX", self.target, "CLEAN_TEXT", "HDBSCAN_CLUSTER", "CAH_CLUSTER", "FINAL_CLUSTER"]]\
                                            .groupby("INDEX")\
                                            .aggregate({
                                                self.target : lambda x : list(x)[0],
                                                "CLEAN_TEXT" : list,
                                                "HDBSCAN_CLUSTER" : list, 
                                                "CAH_CLUSTER" : list, 
                                                "FINAL_CLUSTER" : list})\
                                            .reset_index()
        output.index = output["INDEX"]

        return output["FINAL_CLUSTER"]


    def manual_fit(self, mapping_dict):

        # make sure all labels have a mapping value
        labels = set(self.data["FINAL_CLUSTER"].unique())
        grouped_labels = []
        new_mapping_dict = {}

        for k, v in mapping_dict.items():
            grouped_labels += v
            for label_id in v:
                new_mapping_dict[label_id] = k 

        labels = labels - set(grouped_labels)
        for i in labels:
            new_mapping_dict[i] = i
        
        self.data["MANUAL_CLUSTER"] =  self.data["FINAL_CLUSTER"].map(new_mapping_dict)
        self.progress("Manual cluster included", 70)

        clusters = self.create_output(cluster_method = "MANUAL_CLUSTER")
        self.progress("Results saved!", 100)

        return clusters


    def predict(self, text):

        # clean text first 
        new_data = self.clean_text(data=text)

        # embedding 
        new_embeddings = self.create_embedding(new_data["CLEAN_TEXT"].tolist())

        for i, m in enumerate(self.classif_models):
            if i == 0:
                preds = m.predict_proba(new_embeddings)
            else:
                preds = preds + m.predict_proba(new_embeddings)
        preds = preds/5

        new_data["PROBA"] = preds.max(axis=1)
        new_data["LABEL"] = preds.argmax(axis=1)
        new_data["LABEL"] = np.where(text["PROBA"] < 0.975, -1, new_data["LABEL"])

        self.output = new_data[["INDEX", "TARGET", "CLEAN_TEXT", "LABEL"]]\
                                            .groupby("INDEX")\
                                            .aggregate({
                                                "CLEAN_TEXT" : list,
                                                "LABEL" : list, 
                                                "TARGET" : lambda x : list(x)[0]})\
                                            .reset_index()

        return self.output["LABEL"]


   