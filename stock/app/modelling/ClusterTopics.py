import umap.umap_ as umap
import hdbscan
from kneed import KneeLocator
import smoothfit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import pandas as pd 
from typing import List
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA
import seaborn as sns
sns.set_theme(style="darkgrid")


class TopicClustering(object):

    def __init__(self, params={}):

        self.params = params 

        self.sbert_model = SentenceTransformer('stsb-mpnet-base-v2')
        self.sbert_model.max_seq_length = 200
        self.bert_embeddings = None


    def create_embedding(self, text : List[str]) -> np.array:
        """
        Bert pre trained model giving embeddings for each sentence

        Args:
            text (List[str]): [input text]

        Returns:
            np.array: [array of embeddings]
        """
        return self.sbert_model.encode(text, show_progress_bar=True, batch_size = 32)


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
        
        #first PCA -> keeps 90% information
        pca = PCA(random_state=42, n_components=0.9)
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

        # embedding 
        print("BERT EMBEDDING")
        bert_embeddings = self.create_embedding(text)
        if self.bert_embeddings is None:
            self.bert_embeddings = bert_embeddings

        # PCA : reduce embeddings dim 
        print("PCA")
        new_embeddings = self.pca_embeddings(bert_embeddings)

        # UMAP : reduce dimmension based on kullback lieber distance 
        if umap:    
            print("UMAP EMBEDDING")
            new_embeddings = self.create_umap(new_embeddings)

        return new_embeddings


    def visualize_cluster(self, embeddings : np.array, cluster : List) -> None:
        """Display a 2 dimensional vision of cluster 

        Args:
            embeddings (np.array): [embeddings array]
            cluster (List): [cluster labels]
        """

        print("----- VISUALIZE CLUSTERING -----")

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


    def c_tf_idf(self, documents : List[str], m : int, ngram_range : tuple):

        count = CountVectorizer(ngram_range=ngram_range).fit(documents)

        t = count.transform(documents).toarray()
        w = t.sum(axis=1)
        tf = np.divide(t.T, w)
        sum_t = t.sum(axis=0)
        idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
        tf_idf = np.multiply(tf, idf)

        return tf_idf, count


    def extract_top_n_words_per_topic(self, docs_per_topic : pd.DataFrame, 
                                            cluster_name : str):
        """
        Extract top n words for each cluster ID made by cluster_name trategy

        Args:
            docs_per_topic ([type]): [dataframe with all comments per row and clustered output so far]
            cluster_name ([type]): [cluster strategy as str]

        Returns:
            [pd.DataFrame, Dict]: [top words as paragraph in dataframe, dictionnary of top n words per cluster ID]
        """
        
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


    def textual_clusters(self, data : pd.DataFrame, 
                            cluster : List[str], cluster_name : str):
        """
        Aggregate all sentence from the same cluster into one paragraph. 
        Create a dataframe compiling first clustering results

        Args:
            data (pd.DataFrame): [input text dataframe]
            cluster (List[str]): [first cluster try]
            cluster_name (str): [clustering method]

        Returns:
            List[pd.DataFrame, pd.DataFrame]: [pari of dataframes]
        """

        if not isinstance(data, pd.DataFrame):
            docs_df = pd.DataFrame(data, columns = ["CLEAN_TEXT"])
        else:
            docs_df = data 

        docs_df[cluster_name] = cluster
        docs_df['DOC_ID'] = range(len(docs_df))
        docs_per_topic = docs_df.groupby([cluster_name], as_index = False).agg({"CLEAN_TEXT": ' '.join})
        
        return docs_df, docs_per_topic


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

     
    def create_embeddings_cluster(self, corpus_text_cluster : pd.DataFrame) -> np.array:
        """
        Create embeddings for each cluster from HDBSCAN. 
        Look at the top n ngrams form each cluster, embed each top n words and take the median 
        for each hdbscan cluster. This will give the barycentre embedding for each cluster.

        Args:
            corpus_text_cluster (pd.DataFrame): [dataframe with one row per cluster]

        Returns:
            pd.DataFrame: [embeddings of barycentre of each cluster] 
        """

        embeddings_cluster = []
        for cl in corpus_text_cluster["HDBSCAN_CLUSTER"].unique():
            cl_index = list(corpus_text_cluster.loc[corpus_text_cluster["HDBSCAN_CLUSTER"] == cl, "TOP_WORDS"].values[0])
            embeddings_cluster += cl_index
        
        embed = self.embedding_reduction(embeddings_cluster, umap=False)

        final_embeddings = []
        for i in range(len(corpus_text_cluster["HDBSCAN_CLUSTER"].unique())):
            final_embeddings.append(np.median(embed[i*self.params["n_words_cluster"]:(i+1)*self.params["n_words_cluster"]], axis=0))

        return np.array(final_embeddings)


    def hdbscan_clustering(self, embeddings : np.array) -> List[int]:
        """
        HDBSCAN clustering on text embedded

        Args:
            embeddings (np.array): [embedding of text from bert pre trained model]

        Returns:
            List[int]: [labels of cluster]
        """

        dbscan = hdbscan.HDBSCAN(min_cluster_size=self.params["min_cluster_size"],
                                min_samples= self.params["min_cluster_size"],
                                metric='euclidean',                      
                                cluster_selection_method='leaf',
                                algorithm='best', 
                                alpha=1.0, 
                                core_dist_n_jobs = multiprocessing.cpu_count() -1)
        cluster = dbscan.fit(embeddings)

        print("----FINISHED HDBSCAN ALGO-----")

        return cluster.labels_


    def CAH_clustering(self, 
                        df : pd.DataFrame, 
                        corpus_text_cluster : pd.DataFrame) -> List[int]:
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

        # embeddings of top n words from each cluster
        embedding_clusters = self.create_embeddings_cluster(corpus_text_cluster)
        pairs = self.calculate_similarity(embedding_clusters, embedding_clusters)

        #to transform similarity to distance
        if pairs.shape[0] == pairs.shape[1]:
            pairs = np.abs(1 - self.cluster_corr(pairs))
        else:
            pairs = np.abs(1 - pairs)

        # plot similarities
        if self.params["verbose"] > 1:
            plt.figure(figsize=(10,10))
            plt.imshow(pairs, cmap='hot', interpolation='nearest')
            plt.title("HDBSCAN cluster similarities")
            plt.show()

        # hierarchical clustering based on similarity distance 
        distArray = ssd.squareform(pairs.round(2))  # scipy converts matrix to 1d array
        hier = sch.linkage(distArray, method="ward")  # you can use other methods 

        # deduce threshold cah
        self.params["threshold_cah"] = self.deduce_threshold_elbow(hier)

        corpus_text_cluster["CAH_CLUSTER"] = sch.fcluster(hier, self.params["threshold_cah"], criterion="distance")
        mapping_clusters = corpus_text_cluster[["HDBSCAN_CLUSTER", "CAH_CLUSTER"]].set_index("HDBSCAN_CLUSTER").to_dict()
        df["CAH_CLUSTER"] = df["HDBSCAN_CLUSTER"].map(mapping_clusters["CAH_CLUSTER"])
        df["CAH_CLUSTER"] = np.where(df["HDBSCAN_CLUSTER"] == -1, -1, df["CAH_CLUSTER"])

        if self.params["verbose"] > 0:
            plt.figure(figsize=(10,10))
            sch.dendrogram(hier, truncate_mode="level", p=40, color_threshold=self.params["threshold_cah"])
            plt.title(f"CAH cluster similarities based on threshold {self.params['threshold_cah']}")
            plt.show()

        return df["CAH_CLUSTER"].tolist()


    def deduce_threshold_elbow(self, hier):

        x = []
        y = []
        for thre in range(0, 400):
            thre = thre/100
            x.append(thre)
            y.append(len(set(sch.fcluster(hier, thre, criterion="distance"))))

        basis, coeffs = smoothfit.fit1d(np.array(x), np.array(y), 0, 4, len(x), degree=1, lmbda=1)
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

        return inflexion*0.95 # max(0.6, (inflexion + elbow)/2 - 0.1)


    def deduce_threshold_dichotomie(self, hier, corpus_text_cluster, df):

        def get_concentration(loc_ap_priori):
            agg = loc_ap_priori[["TARGET", "CAH_CLUSTER"]].groupby("TARGET").aggregate({"CAH_CLUSTER" : lambda x : set(x) - set([-1])})
            return agg["CAH_CLUSTER"].apply(lambda x : len(x)).mean()

        upper_thre = 3.5
        lower_thre = 0.1
        self.params["threshold_cah"] = 1

        loc_ap_priori = df.loc[df["INDEX"] == -1]
        loc_ap_priori["CAH_CLUSTER"] = 1

        # dichotomie to find best threshold clustering all a priori key words
        while abs(upper_thre - lower_thre) > 0.01:
            volume = get_concentration(loc_ap_priori)
            # logging.info(f"THRESHOLD : {self.params['threshold_cah']} / AVG CLUSTER PER CATEGORY : {volume}")

            corpus_text_cluster["CAH_CLUSTER"] = sch.fcluster(hier, self.params["threshold_cah"], criterion="distance")
            mapping_clusters = corpus_text_cluster[["HDBSCAN_CLUSTER", "CAH_CLUSTER"]].set_index("HDBSCAN_CLUSTER").to_dict()
            loc_ap_priori["CAH_CLUSTER"] = loc_ap_priori["HDBSCAN_CLUSTER"].map(mapping_clusters["CAH_CLUSTER"])
            loc_ap_priori["CAH_CLUSTER"] = np.where(loc_ap_priori["HDBSCAN_CLUSTER"] == -1, -1, loc_ap_priori["CAH_CLUSTER"])

            if get_concentration(loc_ap_priori) <= 1.5:
                upper_thre = self.params["threshold_cah"]
                self.params["threshold_cah"] = upper_thre - (upper_thre - lower_thre)/2
            else:
                lower_thre = self.params["threshold_cah"]
                self.params["threshold_cah"] = lower_thre + (upper_thre - lower_thre)/2

        self.params["threshold_cah"] =  (upper_thre + lower_thre)/2

    
    def select_top_k_sentences_per_cluster(self, cluster_method = "FINAL_CLUSTER", k_sentences=30):

        render = pd.DataFrame([], index= list(range(k_sentences)))

        for id_cluster in np.sort(self.data [cluster_method].unique()):
            if id_cluster!=-1:
                sub_cluster = self.data.loc[self.data [cluster_method] == id_cluster]
                sub_cluster = sub_cluster.loc[sub_cluster["_TEXT_COMMENT"].apply(lambda x : len(str(x)))> 35]
                sub_index = list(sub_cluster.index) 

                similarities = self.calculate_similarity(self.bert_embeddings[sub_index], self.bert_embeddings[sub_index])
                sub_cluster["SIMILARITIES"] = similarities.sum(axis=0)
                sub_cluster = sub_cluster.sort_values("SIMILARITIES", ascending=0)

                # per step 
                # if k_sentences < sub_cluster.shape[0]:
                #     step = int(sub_cluster.shape[0]/k_sentences)
                # else:
                #     step = 1

                # indexes = []
                # for i in range(k_sentences):
                #     indexes.append(sub_index[i*step])

                # closest sentences to all and furthest
                indexes = sub_index[-k_sentences:]
                response =  sub_cluster.loc[indexes, "_TEXT_COMMENT"].tolist()

                if len(response) < k_sentences:
                    response = response + [""]*(k_sentences - len(response))

                render[f"CLUSTER_{id_cluster}"] = response
        
        return render
