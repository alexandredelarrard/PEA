import pandas as pd
from io import BytesIO
import plotly.graph_objs as go
from datetime import datetime
import plotly.express as px


class AnalysisTopicExtraction(object):
    
    def __init__(self, target=""):

        self.target= target
        self.seed = 42 
        self.today = datetime.today().strftime("%Y-%m-%d")


    def analyse(self, df, n_words, cluster_method="FINAL_CLUSTER"):

        #remove seed words if any
        df = df.loc[df["INDEX"] != -1]
        
        top_words = {}
        for k, v in n_words.items():
            top_words[k] = list(zip(*v[:2]))[0]

        confidence = {}
        for k, v in n_words.items():
            confidence[k] = sum(list(zip(*v[:5]))[1])

        df["TOP_WORDS"] = df[cluster_method].map(top_words)
        df_bar_chart = df[["TOP_WORDS", cluster_method]].groupby([cluster_method]).count().reset_index()
        df_bar_chart["WORDS"] =  df_bar_chart[cluster_method].map(top_words)
        df_bar_chart["CONFIDENCE"] =  df_bar_chart[cluster_method].map(confidence)
        df_bar_chart["CONFIDENCE"] = df_bar_chart["CONFIDENCE"].round(3)
        df_bar_chart = df_bar_chart.loc[df_bar_chart[cluster_method] != -1]

        df_bar_chart = df_bar_chart.sort_values("TOP_WORDS", ascending=1)
        df_bar_chart["TOP_WORDS"] = df_bar_chart["TOP_WORDS"]*100/df_bar_chart["TOP_WORDS"].sum()

        df_bar_chart["WORDS"] = df_bar_chart[["WORDS", cluster_method]].apply(lambda x: str(x[1]) + ": " + " ".join(x[0]), axis=1)

        df_bar_chart.rename(columns={"TOP_WORDS" : "PERCENTAGE OF ALL COMMENTS (%)", "WORDS" : "CLUSTER"}, inplace=True)    

        fig = px.bar(df_bar_chart, x="PERCENTAGE OF ALL COMMENTS (%)", y="CLUSTER", orientation='h',
                    title=self.target,
                    height=1000, width=800, color='CONFIDENCE')

        #to save values
        output = BytesIO()
        writer2 = pd.ExcelWriter(output, engine='xlsxwriter')
        df_bar_chart.to_excel(writer2, sheet_name = 'BAR_CHART', index = False)
        writer2.save()
        processed_data = output.getvalue()
        self.df_bar_chart = processed_data

        return fig


    def clean_top_words(self, words):

        #clean words
        clean_words = {}
        for k, v in words.items():
            if k != -1:
                clean_words[k] = []
                for w in v:
                    clean_words[k].append(w[0])

        clean_words = pd.DataFrame().from_dict(clean_words)

        clean_words.index = ["NGRAM " + str(x) for x in clean_words.index]
        clean_words.columns = ["Cluster " + str(x) for x in clean_words.columns]

        return clean_words


    def create_comments_table(self, df, cluster_method):

        b = df[["_TEXT_COMMENT", cluster_method]].groupby(cluster_method)["_TEXT_COMMENT"].apply(list).reset_index()
        b.index = b[cluster_method]
        b["len"] = b["_TEXT_COMMENT"].apply(lambda x : len(x))
        b["len"] = max(b["len"]) - b["len"]
        b["_TEXT_COMMENT"] = b[["len", "_TEXT_COMMENT"]].apply(lambda x: x[1] + [""]*x[0], axis=1)
        c = b["_TEXT_COMMENT"].to_dict()
        c = pd.DataFrame().from_dict(c)

        # remove no clustered cluster
        del c[-1]
        c.columns = ["CLUSTER " + str(x) for x in c.columns]

        return c


    def create_plotly_df(self, data):

        cols = data.columns      
        values = []
        for col in cols:
            values.append(data[col])

        fig = go.Figure(data=[go.Table(
            columnorder = list(range(1, len(cols)+1)),
            columnwidth = [90]*len(cols),
            header=dict(values=list(cols),
                        line_color='darkslategray',
                        fill_color='royalblue',
                        align='center',
                        font=dict(color='white', size=12),
                        height=40),
            cells=dict(values=values,
                    fill_color='white',
                    line_color='darkslategray',
                    align='left',
                    font_size=12,
                    height=30))
        ])

        return fig



    def analyse_hue(self, df, n_words, hue, cluster_method = "FINAL_CLUSTER"):

        #remove seed words if any
        df = df.loc[df["INDEX"] != -1]

        top_words = {}
        for k, v in n_words.items():
            top_words[k] = list(zip(*v[:2]))[0]

        confidence = {}
        for k, v in n_words.items():
            confidence[k] = sum(list(zip(*v[:5]))[1])

        df[hue] = df[hue].astype(str)

        df["TOP_WORDS"] = df[cluster_method].map(top_words)
        df_bar_chart = df[["TOP_WORDS", hue, cluster_method]].groupby([hue, cluster_method]).count().reset_index()
        df_bar_chart["WORDS"] =  df_bar_chart[cluster_method].map(top_words)
        df_bar_chart["CONFIDENCE"] =  df_bar_chart[cluster_method].map(confidence)
        df_bar_chart["CONFIDENCE"] = df_bar_chart["CONFIDENCE"].round(3)
        df_bar_chart = df_bar_chart.loc[df_bar_chart[cluster_method] != -1]

        df_bar_chart = df_bar_chart.sort_values("TOP_WORDS", ascending=1)

        # normalize per hue volume
        number_hue = df_bar_chart[["TOP_WORDS", hue]].groupby(hue).sum().to_dict()["TOP_WORDS"]
        df_bar_chart["DIVIDER"] = df_bar_chart[hue].map(number_hue)
        df_bar_chart["TOP_WORDS"] = df_bar_chart["TOP_WORDS"]*100/df_bar_chart["DIVIDER"]

        df_bar_chart["WORDS"] = df_bar_chart[["WORDS", cluster_method]].apply(lambda x: str(x[1]) + ": " + " ".join(x[0]), axis=1)

        df_bar_chart.rename(columns={"TOP_WORDS" : "PERCENTAGE OF ALL COMMENTS (%)", "WORDS" : "CLUSTER"}, inplace=True)   

        fig = px.bar(df_bar_chart, x="PERCENTAGE OF ALL COMMENTS (%)", y="CLUSTER", orientation='h',
                    title=self.target,
                    height=1200, width=900,
                    color=hue, barmode="group")

        #to save values
        output = BytesIO()
        writer2 = pd.ExcelWriter(output, engine='xlsxwriter')
        df_bar_chart.to_excel(writer2, sheet_name = 'BAR_CHART', index = False)
        writer2.save()
        processed_data = output.getvalue()
        self.df_bar_chart_hue = processed_data

        return fig


    def save_all(self, df, words, to_fix, cluster_method="FINAL_CLUSTER"):

        to_fix = to_fix["ORIGIN"].value_counts().reset_index()

        output = BytesIO()
        writer2 = pd.ExcelWriter(output, engine='xlsxwriter')
       
        c = self.create_comments_table(df, cluster_method)
        c.to_excel(writer2, sheet_name = 'GROUPED_COMMENTS', index = False)
        df.to_excel(writer2, sheet_name = 'FULL_DATA', index = False)
        to_fix.to_excel(writer2, sheet_name = 'WORDS_TO_CLEAN', index = False)

        clean_words = self.clean_top_words(words)
        clean_words.to_excel(writer2, sheet_name = 'TOP_WORDS', index = False)

        writer2.save()
        processed_data = output.getvalue()

        return processed_data