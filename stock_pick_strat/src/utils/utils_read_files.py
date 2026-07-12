import pandas as pd
from glob import glob
from typing import Dict
from pathlib import Path
import os
from tqdm import tqdm
import pickle
import logging
import json


def read_crawled_csvs(path: Path):

    # read all csvs
    files = glob(str(path / Path("*.csv")))
    not_read = []
    liste_dfs = []

    for file in tqdm(files):
        try:
            df_file = read_csv(file)
            if "FILE" not in df_file.columns:
                df_file["FILE"] = os.path.basename(file)

            liste_dfs.append(df_file)
        except Exception:
            not_read.append(file)

    if len(liste_dfs) != 0:
        df = pd.concat(liste_dfs, axis=0, ignore_index=True)
    else:
        df = pd.DataFrame()

    logging.info(f"RECORDINGS : {df.shape[0]}")
    logging.info(f"Missing reads of files : {len(not_read)}")

    return df


def read_csv(file, sep=";"):
    df_file = pd.read_csv(file, sep=sep)
    return df_file


def read_pickle(path: Path):
    with open(path, "rb") as f:
        df_file = pickle.load(f, encoding="latin-1")
    return df_file


def read_json(path: str):
    with open(path, "r") as f:
        df_file = json.load(f)
    return df_file


def save_json(dico, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dico, f, ensure_ascii=False)


def save_pickle_file(df, path):
    with open(path, "wb") as f:
        pickle.dump(df, f)


def save_queue_to_file(queue, path: Path):
    infos = []
    while queue.qsize() != 0:
        infos.append(queue.get())
    save_infos(infos, path)


def save_infos(df: pd.DataFrame, path: Path):

    _, file_extension = os.path.splitext(path)

    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))

    if file_extension == ".csv":
        df.to_csv(path, index=False, sep=";")
    elif file_extension == ".txt" or file_extension == ".pickle":
        with open(path, "wb") as f:
            pickle.dump(df, f)
    else:
        logging.error(
            f"Extensions handled for saving files are .TXT / .PICKLE or .CSV only. Found {file_extension}"
        )


def keep_files_to_do(to_crawl, already_crawled):
    liste_urls = list(set(to_crawl) - set(already_crawled))
    logging.info(
        f"ALREADY CRAWLED {len(already_crawled)} REMAINING {len(liste_urls)} / {len(to_crawl)}"
    )
    return liste_urls


def check_path_exist(path):
    path = Path(path)
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)


def read_local_csv_data(path: str, sep=",") -> Dict:
    files = glob(path + "/*.csv")
    dict_local = {}
    for f in files:
        base_name = os.path.basename(f).replace(".csv", "")
        dict_local[base_name] = read_csv(f, sep=sep)
    return dict_local
