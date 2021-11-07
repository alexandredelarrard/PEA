import pandas as pd
import numpy as np
import re
from typing import List
import contractions
import logging
import tqdm
import swifter 
import spacy

nlp = spacy.load("en_core_web_lg")

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

import pkg_resources
from symspellpy import SymSpell, Verbosity
sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
if sym_spell.word_count:
    pass
else:
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer


class TextCleaner(object):
    """Input text cleaning 
    inputs : 
        - language: text language (for stopwords)
        - remove_words: list of strings for extra stop words to remove 
        - list_tuples_replace : matching list of tuples of words to replace on text - industry specific
          e.g: ("PSG", "Paris Saint Germain")
        - is_sp_removed : boolean : should remove words and replace extra words ? 

    Main cleanings ares :
    - split paragraphs on sentences (could increase the size of the input)
    - remove punctuation 
    - lower case 
    - remove numbers
    - replace industry specific words 
    - check miss spellings for words longer than 5 caracter (english only)
    - replace contraction words (english only)
    - keep noun to noun chunks
    - keep final comments of at least 6 caracters

    Args:
        object ([Array / pd.Series]): [List/Array/Serie of comments]
    """

    def __init__(self, stop_word_list = [], language="english", 
                is_sp_removed=True, remove_words=[], industry_words=pd.DataFrame([])):

        self.language= language
        self.stop_word_list = stop_word_list
        self.is_sp_removed = is_sp_removed
        self.remove_words = remove_words
        self.list_tuples_replace = []
        self.porter = PorterStemmer()

        if industry_words.shape[0] > 0:
            self.list_tuples_replace = list(industry_words.to_records(index=False))

        self.check_language()
        self.get_stop_words()


    def get_stop_words(self):
        """
        Create list of words to remove
        """
        if len(self.stop_word_list) == 0:
            self.stop_word_list = stopwords.words(self.language)

        # remove important stop words from removelist
        self.stop_word_list = self.remove_words + \
                                list(set(self.stop_word_list) - \
                                set(["more", "not", "no", "off", "down"])) + \
                                ["none", "na", "nan", "cr"]


    def check_language(self):
        """
        Check input language is in ["english"] 
        """
        list_languages = ["english"]
        if self.language not in list_languages:
            logging.error(f"[Language] Please ensure stopwords are in {list_languages}")


    def exclude_impossible_answers(self, text, nbr_caracters=4):
        """Exclude comments with less than nbr_caracters caracters

        Args:
            text ([array]): [input text]

        Returns:
            [array]: [text without pathological cases]
        """

        count_len = lambda x : len(str(x))
        vfunc = np.vectorize(count_len)
        length = vfunc(text)
        text = text[length >= nbr_caracters]

        return text


    def replace_nnp(self, x):
        """Input comment as a string to be cleaned up:
        - lower case 
        - contraction handled 
        - punctuation replace as ' '
        - strip and check multi spaces 

        Args:
            x ([str]): [input text]

        Returns:
            [str]: [cleaned input text]
        """

        # normalization 1: xxxThis is a --> xxx. This is a (missing delimiter)
        x = re.sub(r'([a-z])([A-Z])', r'\1\. \2', str(x))  # before lower case

        x = " " + x.lower() + " "
        
        # handle contractions
        x = contractions.fix(x)

        # remove parenthesised text
        x = re.sub(r'\([^)]*\)', ' ', x)

        # handle specific cases
        x = re.sub("/n"," ", x)
        x = re.sub("\n", " ", x)
        x = re.sub("&", " and ", x)
        x = re.sub("@", " at ", x)
        x = re.sub(" etc ", " ", x)
        
        # punctuation removed
        x = re.sub("[^\w\s_]", " ", x)

        #replace 
        x =  str(x).replace(" e g ", " example ")

        # remove extra spaces
        x = re.sub(" + ", " ", x)

        # strip sides 
        x = x.strip()

        return x

        
    def get_chunk(self, x : str) -> str:
        """ Filter out sentences based on noun to noun extraction. 
        Ensure size of string is sufficiently high

        Args:
            x ([str]): [comment / text]

        Returns:
            [type]: [chunked comment / text]
        """

        if len(x) <= 50:
            return x 

        chunks =  list(nlp(str(x)).noun_chunks)

        if len(chunks) == 0:
            return x

        right_chunk = str(chunks[0])
        if len(chunks) > 1:
            for nouns in chunks[1:]:
                right_chunk = right_chunk + " " + str(nouns)

        if len(str(right_chunk)) <= 5:
            return x

        return str(right_chunk)


    def append_data_dico(self, data, dictionnary):

        for k, word in dictionnary.to_records(index=False):
            data = data.append({"TARGET": k, 
                                "_TEXT_COMMENT": word, 
                                "INDEX":-1, 
                                "CLEAN_TEXT": self.replace_nnp(word)}, ignore_index=True)

        return data


    def f_word_replace(self, w_list :  List[str]) -> List[str]:
        """
        replace consulting words
        """
        
        w_list_fixed = []
        for word in w_list:
            for rpl in self.list_tuples_replace:
                if word == rpl[0]:
                    word = rpl[1]
            w_list_fixed.append(word.split(" "))

        return [item for sublist in w_list_fixed for item in sublist]


    def f_stopw(self, w_list : List[str], list_stop_words : List[str]) -> List[str]:
        """
        filtering out stop words
        """
        return [word for word in w_list if word not in list_stop_words]


    def f_typo(self, w_list : List[str]) -> List[str]:
        """
        rewrite miss spelled words and remove one letter ones 
        :param w_list: word list to be processed
        :return: w_list with typo fixed by symspell. words with no match up will be dropped
        """

        w_list_fixed = []
        for word in w_list:
            suggestions = sym_spell.lookup(word, 
                                        Verbosity.CLOSEST, 
                                        max_edit_distance=3, 
                                        include_unknown=False)
            if suggestions:
                if len(suggestions[0].term) > 3 and len(word)> 3:
                    # if  suggestions[0].term != word:
                    #     print(word, suggestions[0].term)
                    w_list_fixed.append(suggestions[0].term)
                else:
                    if len(word)>1:
                        w_list_fixed.append(word)
            else:
                pass

        return w_list_fixed

        
    # filtering out punctuations and numbers
    def f_number(self, w_list : List[str]) -> List[str]:
        """
        :param w_list: word list to be processed
        :return: w_list with punct and number filter out
        """
        return [word for word in w_list if word.isalpha()]


    def stemSentence(self, sentence):
        token_words=word_tokenize(sentence)
        stem_sentence=[]
        for word in token_words:
            stem_sentence.append(self.porter.stem(word))
            stem_sentence.append(" ")
        return "".join(stem_sentence)


    def preprocess_word(self, new_text : np.array, list_stop_words : List[str]) -> np.array:
        cut_tt = []

        logging.info(f"Cleaning stop words / punctuation / miss spelling for {self.language}")
        for txt in tqdm.tqdm(new_text):
            w_list = nltk.word_tokenize(txt)

            # replace specifc industry words
            w_list = self.f_word_replace(w_list)

            # remove stop words
            w_list = self.f_stopw(w_list, list_stop_words)

            #clean miss spelled words / remove unknown
            w_list = self.f_number(w_list)

            #clean miss spelled words / remove unknown
            # w_list = self.f_typo(w_list)

            cut_tt.append(" ".join(w_list))

        new_text = np.array(cut_tt)

        return new_text


    def split_sentences(self, x): 
        sentences = sent_tokenize(x)
        if len(sentences) <= 3:
            return x 
        else: 
            length_sentences = [len(u) for u in sentences]
            threshold = np.percentile(length_sentences, 75)
            index_centences =  length_sentences >= threshold
            return ". ".join((np.array(sentences)[index_centences]))


    def clean_text(self, data, target) -> pd.DataFrame:
        """
        Clean text for class terms in a vectorized way
        """

        # chunk into sentences 
        logging.info(f"Split text into sentences")
        data["TARGET"] = data[target]

        # clean_final_comments
        data["_TEXT_COMMENT"] = data["TARGET"].swifter.set_dask_scheduler(scheduler="processes").apply(lambda x: self.split_sentences(x))
        # data = data.explode("_TEXT_COMMENT")
        # data["_TEXT_COMMENT"] = data["_TEXT_COMMENT"].apply(lambda x : str(x))
        data = data.loc[~data["_TEXT_COMMENT"].isin([" "])]

        # index ref through the code
        data["ID_TEXT"] = data.index
        data = data.reset_index(drop=True) 
        data["INDEX"] = data.index

        # it will be used as a separator 
        logging.info(f"Normalize text")
        vfunc = np.vectorize(self.replace_nnp)
        new_text = vfunc(data["TARGET"].tolist())

        # remove stop words 
        if self.is_sp_removed:
            data["CLEAN_TEXT"] = self.preprocess_word(new_text, self.stop_word_list)

        data["LENGTH_TEXT"] = data["CLEAN_TEXT"].apply(lambda x: len(x))
        long_text = int(np.percentile(data["LENGTH_TEXT"], 90))
        data["CLEAN_TEXT"] = data["CLEAN_TEXT"].apply(lambda x : x[:min(len(x), long_text)])

        # chunk size
        # logging.info(f"Split text into noun chunks and rejoin")
        # df["CLEAN_TEXT"] = df["CLEAN_TEXT"].swifter.set_dask_scheduler(scheduler="processes").apply(lambda x: self.get_chunk(x))
        
        return data
