import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from anytree import Node
from anytree.exporter import DotExporter
from graphviz import render, Source
sns.set_theme(style="darkgrid")

from utils.general_functions import check_save_path


class AnalysisTopicExtraction(object):
    
    def __init__(self, title="", save_path=".", filter_analysis=None):
        
        self.title= title.lower().strip()
        self.title = self.title[:min(len(self.title),15)]
        self.seed = 42 
        self.today = datetime.today().strftime("%Y-%m-%d")
        self.filter_analysis = filter_analysis

        if save_path=="":
            save_path = os.environ["USERPROFILE"]
            logging.warning(f"SAVING OUTPUTS TO {save_path}")
        self.save_path = save_path


    def analyse(self, data, dict_n_words, k_words=4):

        check_save_path(self.save_path)

        df = data.loc[data["INDEX"] != -1]
        for cluster_name, n_words in dict_n_words.items():

            top_words = {-1 : [("", 0)]*k_words}

            for k, v in n_words.items():
                if k != -1:
                    top_words[k] = list(zip(*v[:k_words]))[0]

            df[f"TOP_WORDS_{cluster_name}"] = df[cluster_name].map(top_words)
            df_bar_chart = df[[f"TOP_WORDS_{cluster_name}", cluster_name]].groupby([cluster_name]).count().reset_index()
            df_bar_chart["WORDS"] =  df_bar_chart[cluster_name].map(top_words)
            df_bar_chart = df_bar_chart.loc[df_bar_chart[cluster_name] != -1]

            df_bar_chart = df_bar_chart.sort_values(f"TOP_WORDS_{cluster_name}", ascending=0)

            plt.figure(figsize=(20,20))
            g = sns.barplot(y="WORDS", x=f"TOP_WORDS_{cluster_name}", data=df_bar_chart)
            plt.setp(g.get_xticklabels(), rotation=85)
            plt.title(f"{cluster_name.upper()} : Text volume and top 4 words of cluster")
            plt.savefig(self.save_path + f'/results_{self.today}_{self.title}_{cluster_name}.png', bbox_inches='tight', pad_inches=0)


    def save_all(self, data, dict_n_words):

        check_save_path(self.save_path)

        df = data.loc[data["INDEX"] != -1]

        options = {}
        options['strings_to_formulas'] = False
        options['strings_to_urls'] = False

        writer2 = pd.ExcelWriter(self.save_path + f'/{self.today}_{self.title}.xlsx', engine='xlsxwriter', options=options)

        df.to_excel(writer2, sheet_name = 'FULL_DATA', index = False)

        for cluster_name, n_words in dict_n_words.items():
            b = df[["_TEXT_COMMENT", cluster_name]].groupby(cluster_name)["_TEXT_COMMENT"].apply(list).reset_index()
            b.index = b[cluster_name]
            b["len"] = b["_TEXT_COMMENT"].apply(lambda x : len(x))
            b["len"] = max(b["len"]) - b["len"]
            b["_TEXT_COMMENT"] = b[["len", "_TEXT_COMMENT"]].apply(lambda x: x[1] + [""]*x[0], axis=1)
            c = b["_TEXT_COMMENT"].to_dict()

            c = pd.DataFrame().from_dict(c)
            c.to_excel(writer2, sheet_name = f'TEXT_CLUSTER_{cluster_name.upper()}', index = False)
        
            words = pd.DataFrame().from_dict(n_words)
            words.columns = c.columns
            words.to_excel(writer2, sheet_name = f'TOP_WORDS_{cluster_name.upper()}', index = False)

        writer2.save()
        writer2.close()


    def build_tree_structure(self, data, 
                            dict_n_words, 
                            tf_idf_words_hdbscan, 
                            hdbscan_top_articles, 
                            top_words=5, 
                            display_top_articles=False):

        check_save_path(self.save_path)
        data = data.copy()

        total_confidence = []

        if self.filter_analysis:
            normalization_const = data.loc[(data[f"HDBSCAN_CLUSTER"] !=-1)&(data[self.filter_analysis])].shape[0] /data.loc[data[f"HDBSCAN_CLUSTER"] !=-1].shape[0]

        cah = len([x for x in data.columns if "CAH_CLUSTER" in x])
        cah_range = list(2**np.arange(1, cah +1))

        leaves = {"Level_0": Node(f"{self.title} \n Volume = {data.shape[0]}", color="black", fillcolor="white")}
        for i, level in enumerate(cah_range):
            for k in data[f"CAH_CLUSTER_{level}"].value_counts().index:
                if k != -1:
                    color = "black"
                    information = ""
                    volume_leaf = data.loc[data[f"CAH_CLUSTER_{level}"] ==k].shape[0]

                    if int(level/2) > 1:
                        parent = data.loc[data[f"CAH_CLUSTER_{level}"] ==k, f"CAH_CLUSTER_{int(level/2)}"].values[0] 
                        parent = "_" + str(parent)  
                    else:
                        parent = ""       

                    for w in range(2):
                        information += f"NGRAMS {w} = " + dict_n_words[f"CAH_CLUSTER_{level}"][k][w][0] + "\n" 
                    
                    information += "VOLUME = " + str(volume_leaf)

                    if self.filter_analysis:
                        volume_filter = data.loc[(data[f"CAH_CLUSTER_{level}"] ==k)&(data[self.filter_analysis])].shape[0]
                        information = information + "\n" + "FILTER : " + \
                                    str(volume_filter)

                        if volume_filter / volume_leaf >= normalization_const:
                            color="green"
                        else:
                            color="red"

                    leaves[f"Level_{i+1}_{k}"] = Node(information, 
                                                parent=leaves[f"Level_{i}{parent}"],
                                                color=color, 
                                                height=1)

        # bottom cluster : HDBSCAN construction of infos
        i= i+1
        for k in data[f"HDBSCAN_CLUSTER"].value_counts().index:
            if k != -1:
                color = "black"
                information = ""
                parent = data.loc[data[f"HDBSCAN_CLUSTER"] ==k, f"CAH_CLUSTER_{level}"].values[0] 
                volume_leaf = data.loc[data["HDBSCAN_CLUSTER"] == k].shape[0]
                confidence = np.mean(list(zip(*dict_n_words["HDBSCAN_CLUSTER"][k]))[1])
                total_confidence.append(confidence)

                for w in range(top_words):
                    information += f"NGRAMS {w} = " + dict_n_words["HDBSCAN_CLUSTER"][k][w][0] + "\n"

                information += "\n"
                for w in range(top_words):
                    information += f"TF-IDF {w} = " + tf_idf_words_hdbscan[k][w][0] + "\n"
                    
                information += "\n"
                information +=  "VOLUME = " + str(volume_leaf) + "\n"
                information += "DISTANCE = " + str(confidence) 

                if self.filter_analysis:
                    volume_filter = data.loc[(data["HDBSCAN_CLUSTER"] ==k)&(data[self.filter_analysis])].shape[0]
                    information = information + "\n" + "FILTER : " + str(volume_filter)

                    if volume_filter / volume_leaf >= normalization_const:
                        color="green"
                    else:
                        color="red"

                leaves[f"Level_{i+1}_{k}"] = Node(information, 
                                                parent=leaves[f"Level_{i}_{parent}"],
                                                color=color, 
                                                height=1)

        # map articles to bottom clusters
        if display_top_articles:
            i= i+1
            for k in data["HDBSCAN_CLUSTER"].value_counts().index:
                if k != -1:
                    color = "black"
                    for k_article in hdbscan_top_articles[k].keys():

                        information = ""
                        for spl in range(0, len(hdbscan_top_articles[k][k_article]), 65):
                            information +=  hdbscan_top_articles[k][k_article][spl:spl + 65] + " \n "

                        leaves[f"Level_{i+1}_{k}"] = Node(information, 
                                                        parent=leaves[f"Level_{i}_{k}"],
                                                        color=color,
                                                        height=1)

        if self.filter_analysis == None:
            tree_save = self.save_path + f'/udo_{self.today}_{self.title}.dot'
        else:
            tree_save = self.save_path + f'/udo_{self.today}_{self.title}_{self.filter_analysis}.dot'

        DotExporter(leaves["Level_0"],
                    nodeattrfunc=lambda node: f"shape=box, style=bold, color={node.color}").to_dotfile(tree_save)
        Source.from_file(tree_save)

        if os.path.exists(tree_save.replace(".dot",".png")):
            os.remove(tree_save.replace(".dot",".png"))
            
        render('dot', 'png', tree_save) 
        os.remove(tree_save)

        print(f"TOTAL CONFIDENCE SCORE = {np.mean(total_confidence)}")
        print(f"NUMBER of HDBSCAN CLUSTER = {len(data['HDBSCAN_CLUSTER'].value_counts().index)}")
                

    def analyse_hue(self, df, n_words, hue):

        check_save_path(self.save_path)

        k_words = 4
        top_words = {-1 : [("", 0)]*k_words}
        for k, v in n_words.items():
            if k != -1:
                top_words[k] = list(zip(*v[:k_words]))[0]

        cluster_method = self.latest_cluster

        # final analysis 
        df["TOP_WORDS"] = df[cluster_method].map(top_words)
        df_bar_chart = df[["TOP_WORDS", hue, cluster_method]].groupby([hue, cluster_method]).count().reset_index()
        df_bar_chart["WORDS"] =  df_bar_chart[cluster_method].map(top_words)
        df_bar_chart = df_bar_chart.loc[df_bar_chart[cluster_method] != -1]

        coefs = 20000/df_bar_chart[[hue, "TOP_WORDS"]].groupby(hue).sum()
        df_bar_chart["COEFS"] = df_bar_chart["YEAR"].map(coefs["TOP_WORDS"])
        df_bar_chart["TOP_WORDS"] = df_bar_chart["TOP_WORDS"]*df_bar_chart["COEFS"]
        df_bar_chart = df_bar_chart.sort_values("TOP_WORDS", ascending=0)

        plt.figure(figsize=(20,10))
        g = sns.barplot(x="WORDS", y="TOP_WORDS", hue=hue, data=df_bar_chart)
        plt.setp(g.get_xticklabels(), rotation=85)
        plt.title(self.title)
        plt.show()