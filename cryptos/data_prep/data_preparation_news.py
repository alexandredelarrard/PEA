import pandas as pd 
import numpy as np
import glob
from transformers import pipeline
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import swifter
import re
from tqdm import tqdm

class PrepareNews(object):

    def __init__(self):
        self.news_path = "./data/history/news"

        
    def load_downloaded_articles(self):

        articles = glob.glob(self.news_path + "/*.csv")

        all = pd.DataFrame()
        for f in articles:
            all = pd.concat([all, pd.read_csv(f)])
        
        return all
    
        
    def clean_articles(self, already_there):

        already_there = already_there.drop_duplicates("url")

        already_there = already_there.fillna("")
        already_there["result"] = already_there["text"] + already_there["result"]
        already_there = already_there.drop(["Unnamed: 0", "text"], axis=1)

        already_there["date"] = pd.to_datetime(already_there["date"], format="%Y/%m/%d")
        already_there = already_there.sort_values("date", ascending = 0)

        # analysis 
        # get title 
        already_there["title"] = already_there["result"].apply(lambda x : " ".join(x.split("\n")[1:3]))
        already_there["type_2"] = already_there["result"].apply(lambda x : x.split("\n")[0])
        already_there["about"] = already_there["result"].apply(lambda x : " ".join(x.split("\n")[-2:]))

        return already_there
    

    def sentiment_scoring(self, already_there):

        def handle_score(text):
            answer = sentiment_task(text)[0]

            if answer["label"] == "negative":
                return -1*answer["score"]
            elif answer["label"] == "positive":
                return answer['score']
            else:
                return 0

        MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        sentiment_task = pipeline("sentiment-analysis", model=MODEL, tokenizer=tokenizer)

        tqdm.pandas()
        already_there["sentiment"] = already_there["title"].apply(lambda x : handle_score(x))
        

    def get_hour_edited(self, already_there):
        already_there["hour"] = already_there["result"].apply(lambda x : re.search(' at (.*?) UTC', x).group(1))
        
        already_there["hour"] = np.where(already_there["hour"] == "14:00", "2:00 pm", 
                                np.where(already_there["hour"] == "15:00", "3:00 pm", 
                                np.where(already_there["hour"] == "08:00", "8:00 am", 
                                np.where(already_there["hour"] == "07:08", "7:08 am", 
                                np.where(already_there["hour"] == "00:47", "1:00 am", 
                                np.where(already_there["hour"] == "9:00", "9:00 am", already_there["hour"]))))))

        already_there["date"] = already_there["date"].dt.strftime("%Y-%m-%d") + " " + already_there["hour"].str.replace(".", "")
        

        already_there["date"] = pd.to_datetime(already_there["date"], format= "%Y-%m-%d %I:%M %p")
        
        return already_there
    

    def main(self):

        already_there = self.load_downloaded_articles()
        already_there = self.clean_articles(already_there)
        already_there = self.sentiment_scoring(already_there)
        already_there = self.get_hour_edited(already_there)
        self.save_news(already_there)


    def save_news(self, already_there):
        latest = already_there["date"].max().strftime("%Y-%m-%d")
        already_there.to_csv(self.news_path + f"/news_until_{latest}.csv", index=False)