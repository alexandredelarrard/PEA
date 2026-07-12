import re
import unidecode
import pandas as pd
import hashlib
from datetime import datetime, timedelta
import ast
from typing import List, Dict
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from src.constants.variables import DATE_FORMAT


def get_best_match(name, reference_list):
    match, score, _ = process.extractOne(
        name, reference_list, scorer=Levenshtein.distance
    )
    return pd.Series([match, score])


def clean_base_name(x):
    x = str(x).lower()
    x = x.replace("none", "")
    x = remove_accents(x)
    x = remove_punctuation(x)
    return x


def homogenize_text(x):
    x = clean_base_name(x)
    x = re.sub(" ", "", x)
    return x


def top_1_homogenize_text(x):
    x = clean_base_name(x)
    x = re.sub(" +", " ", x).strip()
    words = x.split(" ")
    return words[0]


def swaped_homogenize_text(x):
    x = clean_base_name(x)
    x = re.sub(" +", " ", x).strip()
    words = x.split(" ")
    x = " ".join(words[::-1])
    x = re.sub(" ", "", x)
    return x


def homogenize_columns(col_names: List) -> List:
    """Rename columns to upper case and remove space and
    non conventional elements

    Args:
        col_names (List): original column names

    Returns:
        List: clean column names
    """

    new_list = []
    for var in col_names:
        var = str(var).replace("/", " ")  # replace space by underscore
        new_var = re.sub("[ ]+", "_", str(var))  # replace space by underscore
        new_var = remove_accents(new_var)
        new_var = re.sub("[^A-Za-z0-9_]+", "_", new_var)
        new_var = re.sub("_$", "", new_var)  # variable name cannot end with underscore
        new_var = new_var.upper()  # all variables should be upper case
        new_list.append(new_var)
    return new_list


def transform_types(dtype: Dict) -> Dict:
    for i, v in dtype.items():
        try:
            dtype[i] = ast.literal_eval(
                v
            )  # float if v=="float" else ( int if v=="int" else str)
        except ValueError as e:
            print(e)
            pass
    return dtype


def remove_accents(x):
    return unidecode.unidecode(x)


def remove_punctuation(x):
    return re.sub(
        "[^A-Za-z0-9_]+", " ", x
    )  # only alphanumeric characters and underscore are allowed


def flatten_dict(mapping_dict):
    flat_dict = {}
    for key, values in mapping_dict.items():
        for value in values:
            flat_dict[value] = key
    return flat_dict


# utils functions
def clean_useless_text(x):
    x = str(x)
    x = x.replace("Lot Details\n", "")
    x = x.replace("Description\n", "")
    x = x.replace("Authenticity guaranteed", "")
    x = x.replace("Photo non contractuelle", "")
    x = x.replace("No reserve\n", "")
    x = x.replace("DETAILS\n", "")
    return x


def remove_dates_in_parenthesis(x):
    pattern = re.compile(r"\([0-9-]+\)")
    return re.sub(pattern, "", x)


def clean_dimensions(x):
    pattern1 = re.compile(r"(\d+.?\d+[ xX]+\d+.?\d+[ xX]+\d+.?\d+)")
    origin = re.findall(pattern1, x)
    if len(origin) == 1:
        origin = origin[0]
        numbers = origin.lower().split("x")
        if len(numbers) == 3:
            new = f" hauteur: {numbers[0].strip()}; largeur: {numbers[1].strip()}; profondeur: {numbers[2].strip()}"
            return x.replace(origin, new)

    pattern2 = re.compile(r"(\d+.?\d+[ xX]+\d+.?\d+)")
    origin = re.findall(pattern2, x)
    if len(origin) == 1:
        origin = origin[0]
        numbers = origin.lower().split("x")
        if len(numbers) == 2:
            new = f" longueur: {numbers[0].strip()}; largeur: {numbers[1].strip()}"
            return x.replace(origin, new)
    return x


def clean_quantity(x):
    x = re.sub(r"(H[\s.:])[\s.:\d+]", " hauteur ", x, flags=re.I)
    x = re.sub(r"(L[\s.:])[\s.:\d+]", " longueur ", x, flags=re.I)
    x = re.sub(r"(Q[\s.:])[\s.:\d+]", " quantite ", x, flags=re.I)
    return x


def clean_shorten_words(x):

    # List of (pattern, replacement, flags, count) tuples
    substitutions = [
        (r"[\s\d+\s](B)\s", " bouteille ", re.I, 1),
        (" bout. ", " bouteille ", re.I, 1),
        (" bt. ", " bouteille ", re.I, 1),
        (r"(bt)", " bouteille ", re.I, 1),
        (r"(mag)", " magnum ", re.I, 1),
        ("@", "a", 0, 0),  # No flags or count needed
        ("n°", " numéro ", 0, 0),
        (" in. ", " inch ", re.I, 0),
        (" ft. ", " feet ", re.I, 0),
        (" approx. ", " approximativement ", 0, 0),
        (" g. ", " gramme ", re.I, 0),
        (" gr. ", " gramme ", re.I, 0),
        (" diam. ", " diametre ", re.I, 0),
    ]

    # Apply all regex substitutions
    for pattern, replacement, flags, count in substitutions:
        x = re.sub(pattern, replacement, x, flags=flags, count=count)

    # List of string replacements
    string_replacements = {
        "¾": "3/4",
        "¼": "1/4",
        "⅐": "1/7",
        "½": "1/2",
    }

    # Apply all string replacements
    for old, new in string_replacements.items():
        x = x.replace(old, new)

    return x


def remove_spaces(x):
    x = str(x).strip()
    x = re.sub(" +", " ", x)
    return x


def remove_lot_number(x):
    return re.sub(r"^(\d+\. )", "", str(x))


def remove_rdv(x):
    # List of substrings to split on
    split_keywords = [
        "\nEstimate",
        "\nSans rendez-vous",
        "\nCondition Report\nProvenance",
        "Les rapports de condition sont",
        "Le meuble ne peut être vu",
        "A lire attentivement :",
        "voir la suite de la description sur le certificat",
        "Pour enchérir, veuillez consulter la section",
        "Catalogue Note\n",
        "\nAdditional Notices",
        "In response to your inquiry, we are",
        "NOTWITHSTANDING THIS REPORT OR ANY",
        "Dans le cadre de nos activités de ventes aux enchères",
        "Délivrance : sur",
        "Expédition : se",
        "**Please be advised that",
        "\nUne TVA de",
        "Footnotes\n",
        "Saleroom notices\n",
        "CONDITIONS DE VENTE",
        "Les présentes Conditions Générales",
    ]

    # Apply each split in sequence
    for keyword in split_keywords:
        x = str(x).split(keyword)[0]

    return x


def define_end_date(end_date):
    if end_date:
        return pd.to_datetime(end_date, format=DATE_FORMAT)
    else:
        return pd.to_datetime(datetime.today() + timedelta(days=30))


def define_start_date(start_date, history_start_year):
    if start_date:
        return pd.to_datetime(start_date, format=DATE_FORMAT)
    else:
        return pd.to_datetime(history_start_year, format="%Y")


def get_number_from_text(text: str):
    match = re.findall("\\d+", text)
    if len(match) > 0:
        return match[0]
    else:
        return None


def encode_file_name(x):
    return hashlib.sha256(str.encode(x)).hexdigest()


def get_in_kwargs(variable: str, kwargs: dict):
    if variable in kwargs.keys():
        return kwargs[variable]
    else:
        raise Exception(f"Should have passed {variable} in kwargs, got {kwargs}")
