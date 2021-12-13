import umap.umap_ as umap
import re
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import pandas as pd 
from typing import List
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import logging
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA
from kneed import KneeLocator
import smoothfit
import seaborn as sns
sns.set_theme(style="darkgrid")

from utils.plot_analysis import plot_dendrogram

class TopicClustering(object):

    def __init__(self, params={}):
        """Use weights from 

        - "all-mpnet-base-v2" -> any subject is the best in 15/10/21 
        - 'allenai/scibert_scivocab_uncased' -> for scientific matters in 15/10/21 

        Args:
            params (dict, optional): [description]. Defaults to {}.
        """

        self.params = params 

        self.sbert_model = SentenceTransformer('all-mpnet-base-v2')
        self.bert_embeddings = None


    def create_embedding(self, text : List[str]) -> np.array:
        """
        Bert pre trained model giving embeddings for each sentence

        Args:
            text (List[str]): [input text]

        Returns:
            np.array: [array of embeddings]
        """
        encode = self.sbert_model.encode(text, show_progress_bar=True, batch_size = 64)
        return encode


    def calculate_similarity(self, text1 : np.array, text2 : np.array) -> pd.DataFrame:
        """
        Calculate cosine distance between two embedded sentences

        Args:
            text1 (np.array): [embedding1]
            text2 (np.array): [embedding2]

        Returns:
            pd.DataFrame: [similarity matrix]
        """
        cosine_scores =  util.pytorch_cos_sim(text1, text2)
        pairs = cosine_scores.detach().cpu().numpy()
        
        return pairs


    def pca_embeddings(self, embeddings : np.array) -> np.array:
        """
        Reduction of embedding dimension with PCA keeping 90% of info

        Returns:
            [Array]: [embeddings]
        """

        # normalize embeddings before pca
        scaler = preprocessing.StandardScaler()
        scaled_embeddings = scaler.fit_transform(embeddings)
        
        #first PCA -> keeps 95% information
        pca = PCA(random_state=42, n_components=0.95)
        af = pca.fit(scaled_embeddings)

        new_embeddings = af.transform(scaled_embeddings)
        return new_embeddings


    def create_umap(self, embeddings : np.array) -> np.array:
        """
        Reduction of embedding dimension with UMAP 

        Returns:
            [Array]: [embeddings]
        """

        if len(embeddings.shape) == 2:
            umap_embeddings = umap.UMAP(random_state=42,
                                        n_neighbors=self.params["umap_n_neighbors"], 
                                        n_components=self.params["umap_n_components"], 
                                        metric='cosine').fit_transform(embeddings)
        else:
            raise Exception("Embedding should have a 2 shaped matrix")

        return umap_embeddings


    def embedding_reduction(self, text : List[str], umap=True) -> np.array:

        # focus on unique words / ngrams 
        if self.params["tf_idf_pre_filtering"]:
            logging.info("FILTER WORDS")
            vectorizer = TfidfVectorizer(ngram_range=(1,1))
            X = vectorizer.fit_transform(text)
            idf = vectorizer.idf_
            words_importance= pd.DataFrame([vectorizer.get_feature_names(), idf]).T.sort_values(1, ascending=0)

            # remove most occuring words and least occuring words
            threshold_too_many = np.percentile(words_importance[1], 2.5)
            threshold_too_few = np.percentile(words_importance[1], 97.5)
            to_remove = words_importance.loc[(words_importance[1] <= threshold_too_many)|(words_importance[1] >= threshold_too_few), 0].tolist()  

            filtered_text = text.apply(lambda x : " ".join([word for word in x.split() if word not in to_remove]))
        
        else:
            filtered_text = text

        # embedding 
        logging.info("BERT EMBEDDING")
        bert_embeddings = self.create_embedding(filtered_text)
        if self.bert_embeddings is None:
            self.bert_embeddings = bert_embeddings

        # PCA : reduce embeddings dim 
        logging.info("PCA")
        new_embeddings = self.pca_embeddings(bert_embeddings)

        # UMAP : reduce dimmension based on kullback lieber distance 
        if umap:    
            logging.info("UMAP EMBEDDING")
            new_embeddings = self.create_umap(new_embeddings)

        return new_embeddings


    def visualize_cluster(self, embeddings : np.array, cluster : List) -> None:
        """Display a 2 dimensional vision of cluster 

        Args:
            embeddings (np.array): [embeddings array]
            cluster (List): [cluster labels]
        """

        logging.info("----- VISUALIZE CLUSTERING -----")

        # Prepare data
        umap_data = umap.UMAP(n_neighbors=15, n_components=2, 
                                min_dist=0.0, metric='cosine').fit_transform(embeddings)
        result = pd.DataFrame(umap_data, columns=['x', 'y'])
        result['labels'] = cluster

        # Visualize clusters
        fig, ax = plt.subplots(figsize=(15, 15))
        outliers = result.loc[result.labels == -1, :]
        clustered = result.loc[result.labels != -1, :]
        plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
        plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
        plt.colorbar()
        plt.show()


    def c_tf_idf(self, documents : List[str], m : int, ngram_range : tuple):

        count = CountVectorizer(ngram_range=ngram_range, max_df=0.5, min_df=2).fit(documents)

        t = count.transform(documents).toarray()
        w = t.sum(axis=1)
        tf = np.divide(t.T, w)
        sum_t = t.sum(axis=0)
        idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
        tf_idf = np.multiply(tf, idf)

        return tf_idf, count


    def extract_tf_idf_words(self, cluster_name : str):
        """
        Extract top n words for each cluster ID made by cluster_name trategy

        Args:
            docs_per_topic ([type]): [dataframe with all comments per row and clustered output so far]
            cluster_name ([type]): [cluster strategy as str]

        Returns:
            [pd.DataFrame, Dict]: [top words as paragraph in dataframe, dictionnary of top n words per cluster ID]
        """

        docs_per_topic = self.data[["CLEAN_TEXT", cluster_name]].groupby([cluster_name], as_index = False).agg({"CLEAN_TEXT": ' '.join})
        
        tf_idf, count = self.c_tf_idf(docs_per_topic["CLEAN_TEXT"].values, 
                                        m=docs_per_topic.shape[0], 
                                        ngram_range=self.params["ngram_range"])
        
        words = count.get_feature_names()
        labels = list(docs_per_topic[cluster_name])
        tf_idf_transposed = tf_idf.T
        indices = tf_idf_transposed.argsort()[:, -self.params["n_words_cluster"]:]
        top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
        
        docs_per_topic["TOP_WORDS"] = docs_per_topic[cluster_name].apply(lambda x: list(zip(*top_n_words[x][:self.params["n_words_cluster"]]))[0])

        return docs_per_topic, top_n_words


    def extract_top_n_words(self):
        """
        Extract top n words for each cluster ID made by cluster_name trategy

        Args:
            cluster_name ([type]): [cluster strategy as str]

        Returns:
            [pd.DataFrame, Dict]: [top words as paragraph in dataframe, dictionnary of top n words per cluster ID]
        """

        # get top words for all text
        text = self.data["CLEAN_TEXT"].tolist()

        # stemming text
        # text = text.apply(lambda x : self.stemSentence(x))
        count = CountVectorizer(ngram_range=self.params["ngram_range"], 
                                max_df=0.5,
                                min_df=2).fit(text)
        
        words = count.get_feature_names()
        words = np.array(words)

        return words 
        
        
    def get_closest_words_to_centroid(self, words, cluster_centroid_embeddings, cluster_name):

        # calculate closest words neighbors of cluster centroids 
        neigh = NearestNeighbors(n_neighbors=self.params["n_words_cluster"])
        neigh.fit(self.top_words_embeddings)
        mapping_words = neigh.kneighbors(cluster_centroid_embeddings)

        top_n_words = {-1 : [("",0)]*self.params["n_words_cluster"]}
        labels = list(set(self.data[cluster_name].value_counts().index) - set([-1]))

        centroides_words = []
        for i, cluster_id in enumerate(np.sort(labels)):
            top_n_words[cluster_id] = [(words[mapping_words[1][i][k]], mapping_words[0][i][k]) for k in range(len(mapping_words[1][i]))]
            centroides_words.append(np.mean(self.top_words_embeddings[mapping_words[1][i][:5]], axis=0))
        
        return top_n_words, np.array(centroides_words)

    
    def cluster_centroid_deduction(self, cluster_method): 

        clustered_data = self.data
        centroids = []

        for cluster_id in np.sort(clustered_data[cluster_method].value_counts().index):
            if cluster_id != -1:
                sub_data = clustered_data.loc[clustered_data[cluster_method] == cluster_id, "INDEX"].tolist()
                sub_embeddings = self.bert_embeddings[sub_data]
                centroids.append(np.mean(sub_embeddings, axis=0))

        return np.array(centroids)

    
    def hdbscan_clustering(self, embeddings : np.array) -> List[int]:
        """
        HDBSCAN clustering on text embedded

        Args:
            embeddings (np.array): [embedding of text from bert pre trained model]

        Returns:
            List[int]: [labels of cluster]
        """

        logging.info("#"*50)
        logging.info("|| HDBSCAN CLUSTERING ||")
        logging.info("#"*50)

        dbscan = hdbscan.HDBSCAN(min_cluster_size=self.params["min_cluster_size"],
                                min_samples= self.params["min_samples"],
                                metric='minkowski',  
                                # cluster_selection_epsilon= self.params["cluster_selection_epsilon"],
                                p=2,                    
                                cluster_selection_method='leaf',
                                algorithm='best', 
                                alpha=1.0, 
                                core_dist_n_jobs = multiprocessing.cpu_count() -1)
        cluster = dbscan.fit(embeddings)

        print("----FINISHED HDBSCAN ALGO-----")

        if self.params["verbose"] > 1:
                self.visualize_cluster(embeddings, cluster.labels_)  

        return cluster.labels_


    def CAH_clustering(self, centroids_words_cluster, threshold) -> List[int]:
        """
        Hierarchical clustering based on embeddings of each cluster previously identified. 
        It get current clusters, aggregate their text as one large paragraph and create embeddings 
        from bert pre trained model. 
        Then CAH performed grouping together sub clusters below threshold_cah parameter.

        Args:
            df (pd.DataFrame): [dataframe with text to cluster and previously clustered output]
            corpus_text_cluster (pd.DataFrame): [dataframe with one row per cluster and concatenated 
                                                comments in one paragraph]

        Returns:
            List[int]: [New higher level cluster output]
        """

        model = AgglomerativeClustering(distance_threshold=None, 
                                        n_clusters=threshold,
                                        affinity="euclidean",
                                        linkage="average")
                                        
                                        # affinity="cosine",
                                        # linkage = "average")
        model = model.fit(centroids_words_cluster)
        cah_clusters = model.labels_

        # map cah cluster to hdbscan cluster 
        mapping_clusters = {-1 : -1}
        for i in range(len(cah_clusters)):
            mapping_clusters[i] = cah_clusters[i]

        cluster = self.data["HDBSCAN_CLUSTER"].map(mapping_clusters)

        return cluster, model


    def treebuild_cah_clustering(self, centroids_words_cluster):

        logging.info("#"*50)
        logging.info("|| CAH CLUSTERING ||")
        logging.info("#"*50)

        k=1
        clusters_cah_tree= pd.DataFrame()
        nbr_clusters= len(self.data["HDBSCAN_CLUSTER"].unique()) - 1 # because of -1 cluster

        while 2**k < nbr_clusters:
            clusters_cah_tree[f"CAH_CLUSTER_{2**k}"], model = self.CAH_clustering(centroids_words_cluster, 2**k)
            k+=1

        if self.params["verbose"] > 1:
            model = AgglomerativeClustering(distance_threshold=0, 
                                        n_clusters=None,
                                        affinity="cosine",
                                        linkage = "average")
            model = model.fit(centroids_words_cluster)
            plt.figure(figsize=(10,10))
            plt.title('Hierarchical Clustering Dendrogram')
            # plot the top three levels of the dendrogram
            plot_dendrogram(model, truncate_mode='level', p=16)
            plt.xlabel("Number of points in node (or index of point if no parenthesis).")
            plt.show()

        if self.params["verbose"] > 1:
                self.visualize_cluster(self.embeddings_reduction, clusters_cah_tree[f"CAH_CLUSTER_{2**(k-1)}"])  

        return clusters_cah_tree


    def fishing_missed(self, cluster_centroid_embeddings, cluster_name, distance_to_median= 0):

        print(f"NO CLUSTERIZE : {sum(self.data[cluster_name] == -1)}/{self.data.shape[0]}")

        # those off 
        off = self.data.loc[self.data[cluster_name] == -1][["INDEX", cluster_name]]
        embed = self.bert_embeddings[off["INDEX"]]

        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(cluster_centroid_embeddings)
        mapping_words = neigh.kneighbors(embed)

        off["DISTANCE_CLUSTER"] = list(zip(*mapping_words[0]))[0]
        off["CLOSEST_ID"]  = list(zip(*mapping_words[1]))[0]
        sns.histplot(off["DISTANCE_CLUSTER"])
        plt.show()

        # those clustered
        on = self.data.loc[self.data[cluster_name] != -1][["INDEX", cluster_name]]
        embed_on = self.bert_embeddings[on["INDEX"]]
        mapping_words = neigh.kneighbors(embed_on)

        on["DISTANCE_CLUSTER"] = list(zip(*mapping_words[0]))[0]
        on["CLOSEST_ID"] = list(zip(*mapping_words[1]))[0]
        sns.histplot(on["DISTANCE_CLUSTER"])
        plt.show()

        # metrics 
        keep_on_off = np.percentile(on["DISTANCE_CLUSTER"], 50 + distance_to_median)
        ditch_on_on = np.percentile(off["DISTANCE_CLUSTER"], 95)
        print(f"KEEP below {keep_on_off} for non clustered")
        print(f"DITCH above {ditch_on_on} for already clustered")

        # allocate cluster or -1 to on / off dataframes 
        on["NEW_CLUSTER"] = np.where(on["DISTANCE_CLUSTER"] <= ditch_on_on, on["CLOSEST_ID"], -1)
        off["NEW_CLUSTER"] = np.where(off["DISTANCE_CLUSTER"] <= keep_on_off, off["CLOSEST_ID"], -1)

        # remerge to the overall dataset 
        self.data.loc[on["INDEX"], cluster_name] = on["NEW_CLUSTER"].tolist()
        self.data.loc[off["INDEX"], cluster_name] = off["NEW_CLUSTER"].tolist()

        print(f"FISHING : NO CLUSTERIZE : {sum(self.data[cluster_name] == -1)}/{self.data.shape[0]}")

    
    def post_cluster(self, cluster_name):

        off = self.data.loc[self.data[cluster_name] == -1][["INDEX", cluster_name]]
        embedding_outliers = self.embeddings_reduction[off["INDEX"]]
        
        off["OUTLIERS_CLUSTER"] = self.hdbscan_clustering(embedding_outliers)
        off["OUTLIERS_CLUSTER"] = np.where(off["OUTLIERS_CLUSTER"] == -1, -1, off["OUTLIERS_CLUSTER"] + self.data[cluster_name].max() + 1)

        self.data.loc[self.data[cluster_name] == -1, cluster_name] = off["OUTLIERS_CLUSTER"]

        return self.data


    def closest_text_to_centroid(self, cluster_centroid_embeddings, cluster_name, top_k_text=2):

        df_cluster = self.data[["INDEX", cluster_name, "TARGET"]]
        df_cluster.index = df_cluster["INDEX"]

        neigh = NearestNeighbors(n_neighbors=top_k_text)
        neigh.fit(self.bert_embeddings)
        mapping_words = neigh.kneighbors(cluster_centroid_embeddings)

        mapped_text = pd.DataFrame(index=range(len(df_cluster[cluster_name].unique()) - 1))
        for id_text in range(top_k_text):
            mapped_text[f"CLOSEST_TEXT_{id_text}"] = list(zip(*mapping_words[1]))[id_text]
            mapped_text[f"CLOSEST_TEXT_{id_text}"] = mapped_text[f"CLOSEST_TEXT_{id_text}"].map(df_cluster["TARGET"])
            
        return mapped_text.to_dict(orient="records")

    ##########################
    # OLD FUNCTIONS
    ##########################
    def deduce_threshold_elbow(self, hier):
    
        x = []
        y = []
        for thre in range(0, 300):
            thre = thre/100
            x.append(thre)
            y.append(len(set(sch.fcluster(hier, thre, criterion="distance"))))

        basis, coeffs = smoothfit.fit1d(np.array(x), np.array(y), 0, 3, len(x), degree=1, lmbda=1)
        kneedle = KneeLocator(x, coeffs[1:], S=1.0, curve="convex", direction="decreasing")

        # compute second derivative
        smooth_d2 = np.gradient(np.gradient(coeffs))

        # find switching points
        infls = np.where(np.diff(np.sign(smooth_d2)))[0]

        inflexion = min(infls)/100 
        elbow = kneedle.knee 

        if self.params["verbose"] > 0:
            plt.figure(figsize=(10,10))
            plt.plot(kneedle.x, kneedle.y)
            plt.vlines(inflexion, 0, max(kneedle.y), linestyles="--", color= "blue")
            plt.vlines(elbow, 0, max(kneedle.y), linestyles="--", color= "blue")
            plt.vlines((inflexion + elbow)/2 - 0.1, 0, max(kneedle.y), linestyles="-", color= "red")
            plt.title("Find cah threshold")
            plt.xlabel("Distance in CAH")
            plt.ylabel("Number of clusters")
            plt.show()

        return 0.1 #min(inflexion, 0.4) #max(0.6, (inflexion + elbow)/2 - 0.1)

    def cluster_corr(self, corr_array : np.array, inplace=False) -> np.array:
        """
        Rearranges the correlation matrix, corr_array, so that groups of highly 
        correlated variables are next to eachother 

        Parameters
        ----------
        corr_array : pandas.DataFrame or numpy.ndarray
            a NxN correlation matrix 
            
        Returns
        -------
        pandas.DataFrame or numpy.ndarray
            a NxN correlation matrix with the columns and rows rearranged
        """

        pairwise_distances = sch.distance.pdist(corr_array)
        linkage = sch.linkage(pairwise_distances, method='complete')
        cluster_distance_threshold = pairwise_distances.max()/2
        idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                            criterion='distance')
        idx = np.argsort(idx_to_cluster_array)

        if not inplace:
            corr_array = corr_array.copy()

        if isinstance(corr_array, pd.DataFrame):
            return corr_array.iloc[idx, :].T.iloc[idx, :]

        return corr_array[idx, :][:, idx]
