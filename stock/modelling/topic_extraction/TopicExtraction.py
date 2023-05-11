from tqdm import tqdm
import logging
import pandas as pd
from modelling.topic_extraction.Analysis import AnalysisTopicExtraction
from modelling.topic_extraction.CleanText import TextCleaner
from modelling.topic_extraction.ClusterTopics import TopicClustering

class TopicExtraction(TopicClustering, TextCleaner, AnalysisTopicExtraction):

    def __init__(self, remove_words=[], params={}, industry_words=pd.DataFrame([]), dictionnary=pd.DataFrame([]), results_path = "", title="", filter_analysis=None):

        if results_path == "":
            logging.warning("results will be saved in root base")
            
        self.dictionnary = dictionnary
        self.params = params

        # init super class
        TextCleaner.__init__(self, remove_words=remove_words, industry_words=industry_words)
        TopicClustering.__init__(self, params=self.params)
        AnalysisTopicExtraction.__init__(self, title=title, save_path=results_path, filter_analysis=filter_analysis)

        
    def update_parameters_large(self, params):
        self.params = {"ngram_range" : (2, 3),
                      "random_seed" : 42,
                      "umap_n_neighbors" : 20, # UMAP 
                      "umap_n_components" : 20, # UMAP
                      "min_cluster_size" : 15, # HDBSCAN
                      "min_samples" : 5, # HDBSCAN
                      "cluster_selection_epsilon" : 0, # HDBSCAN
                      "n_words_cluster" : 10, # number of top words to display in excel results / tree = N/2
                      "n_top_text_cluster" : 2, # number of articles to represent per final cluster 
                      "distance_to_median" : 0, # proportion to force to cluster amongst -1 based on average distance to centroid
                      "tf_idf_pre_filtering" : False,
                      "verbose" : 0}

        for key in params.keys():
            self.params.pop(key)

        self.params.update(params)


    def update_parameters_small(self, params):

        self.params = {"ngram_range" : (2, 3),
                      "random_seed" : 42,
                      "umap_n_neighbors" : 30, # UMAP 
                      "umap_n_components" : 30, # UMAP
                      "min_cluster_size" : 5, # HDBSCAN
                      "min_samples" : 10, # HDBSCAN
                      "cluster_selection_epsilon" : 0, # HDBSCAN
                      "n_words_cluster" : 10, # number of top words to display in excel results / tree = N/2
                      "n_top_text_cluster" : 2, # number of articles to represent per final cluster 
                      "distance_to_median" : 10, # proportion to force to cluster amongst -1 based on average distance to centroid
                      "tf_idf_pre_filtering" : False,
                      "verbose" : 0}

        for key in params.keys():
            self.params.pop(key)

        self.params.update(params)

    
    def transformer(self, text, target):

        # clean text first 
        self.data = self.clean_text(text, target)
       
        if self.data.shape[0] > 6000:
            self.update_parameters_large(self.params)
        else:
            self.update_parameters_small(self.params)

        if self.dictionnary.shape[0] > 0:
            self.data = self.append_data_dico(self.data, self.dictionnary)

        # embedding 
        self.embeddings_reduction = self.embedding_reduction(self.data["CLEAN_TEXT"], umap=True)

        self.words = self.extract_top_n_words()
        self.top_words_embeddings = self.create_embedding(self.words)


    def fit(self):

        # CLUSTERING Loop 
        for cluster_name in ["HDBSCAN_CLUSTER", "CAH_CLUSTER"]:

            if cluster_name == "HDBSCAN_CLUSTER":
                self.data[cluster_name] = self.hdbscan_clustering(self.embeddings_reduction)

                # recluster -1 outliers 
                self.data = self.post_cluster(cluster_name)

                # get top n words per cluster based on ngrams 
                self.cluster_centroid_embeddings = self.cluster_centroid_deduction(cluster_name)

                # fishing into -1 if hdbscan (to have all events even multi clustering)
                self.fishing_missed(self.cluster_centroid_embeddings, 
                                    cluster_name, 
                                    distance_to_median=self.params["distance_to_median"])
                self.cluster_centroid_embeddings = self.cluster_centroid_deduction(cluster_name)

                df_top_n_words, centroids_words_cluster = self.get_closest_words_to_centroid(self.words, 
                                                                        self.cluster_centroid_embeddings, 
                                                                        cluster_name)

                docs_per_topic, self.tf_idf_words_hdbscan = self.extract_tf_idf_words(cluster_name)
                self.hdbscan_top_articles = self.closest_text_to_centroid(self.cluster_centroid_embeddings, 
                                                                cluster_name,
                                                                top_k_text=self.params["n_top_text_cluster"])

            if cluster_name == "CAH_CLUSTER":

                # check best : cluster or word centroid ? 
                clusters_cah_tree = self.treebuild_cah_clustering(self.cluster_centroid_embeddings) #centroids_words_cluster
                self.data = pd.concat([self.data, clusters_cah_tree], axis=1)

                # get top n words per cluster based on ngrams 
                self.top_words_cah_clusters = {"HDBSCAN_CLUSTER" : df_top_n_words}
                for cluster_cah_name in tqdm(clusters_cah_tree.columns):
                    self.cluster_centroid_embeddings = self.cluster_centroid_deduction(cluster_cah_name)
                    df_top_n_words, centroids_words_cluster = self.get_closest_words_to_centroid(self.words, 
                                                                            self.cluster_centroid_embeddings, 
                                                                            cluster_cah_name)
                    self.top_words_cah_clusters[cluster_cah_name] = df_top_n_words

        return self.data

    
    def plot_clusters(self, data, display_top_articles=False):
        self.build_tree_structure(data, 
                                    self.top_words_cah_clusters, 
                                    self.tf_idf_words_hdbscan,
                                    self.hdbscan_top_articles,
                                    top_words=int(self.params["n_words_cluster"]/2),
                                    display_top_articles=display_top_articles)
        # self.analyse(data, self.top_words_cah_clusters)
        self.save_all(data, self.top_words_cah_clusters)


    def predict(self, text):

        # clean text first 
        new_data = self.clean_text(data=text)

        # embedding 
        new_embeddings = self.create_embedding(new_data["CLEAN_TEXT"].tolist())

        return 0


   