#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import logging

import pandas as pd


def create_table_preco(db):

    # check if table exists
    query_extists = """
                    SELECT TABLE_NAME
                    FROM information_schema.tables
                    """
    tables = pd.read_sql(query_extists, con=db.engine)

    if "preco" not in tables["TABLE_NAME"].unique():
        query = "CREATE TABLE preco (MACRO_UB_CODE VARCHAR(100) NULL)"
        logging.info("Query to create table preco")
        with db.engine.connect() as connection:
            connection.execute(query)


def transform_python_to_sql_type(dtype):
    if dtype == int:
        return "INT"
    elif dtype == float:
        return "FLOAT"
    elif dtype == str:
        return "VARCHAR(120)"
    elif dtype == pd.datetime:
        return "DATE"
    else:
        logging.warning("only supporting str, int, float, datetime formats to transform to sql !")


def add_columns_in_table_sql(db, table_name, list_columns):

    for columns in list_columns:
        col = columns["name"]
        data_type = transform_python_to_sql_type(columns["type"])
        table = pd.read_sql(f"SELECT TOP 1 * FROM {table_name}", con=db.engine)

        if col not in table.columns:
            query = f"ALTER TABLE {table_name} ADD [{col}] {data_type} NULL"
            logging.info(f"ADDING new column to {table_name}: {col}")
            with db.engine.connect() as connection:
                connection.execute(query)
