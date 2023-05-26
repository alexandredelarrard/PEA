#
# Copyright (c) 2022 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-
import logging
import os

from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL


class SQLServer:
    """
    Declare the database structure of postgres based on server specifics (PWD, NAME)

    db = SQLServer()
    with db as con:
    data = pd.read_sql(query, con=con)
    """

    def __init__(self, host=None, database=None, port=None, user=None, password=None):

        self.database_host = host
        self.cnx = None
        self.engine = None
        self.database = database
        self.database_port = port

        # cached properties
        self._default_user = user
        self._default_password = password

        # Log server information
        logging.info(f"Created SQL server interface with connector {self}")

    def load_config(self):
        self.host = self.load_from_env("DATABASE_HOST")
        self.database = self.load_from_env("DATABASE")
        self.port = self.load_from_env("DATABASE_PORT")

    def __enter__(self):
        """
        Executed when creating a connection to the server

        Returns
        -------
            engine : sqlalchemy engine
        """

        logging.debug(f"Opening connection with {self}")

        self.engine = create_engine(self.connexion_string)
        self.cnx = self.engine.connect()

        return self.engine

    def __exit__(self, exception_type, exception_value, traceback):
        """
        Executed when closing a connection to the server

        Returns
        -------
            cnx : connection to the server
        """
        logging.debug(f"Closing connection with {self}")

        self.cnx.close()

    @property
    def connexion_string(self) -> str:
        """
        Connection string to connect to the DB.
        """

        driver = "ODBC Driver 17 for SQL Server"
        server = os.environ["DATABASE_HOST"]
        database = os.environ["DATABASE"]
        username = os.environ["DATABASE_USER"]
        password = os.environ["DATABASE_PWD"]
        port = "1433"

        url_object = URL.create(
            "mssql+pyodbc",
            username=username,
            password=password,  # plain (unescaped) text
            host=server,
            database=database,
            port=port,
            query=dict(driver=driver),
        )

        return url_object

    def load_from_env(self, parameter):
        prm = ""
        try:
            prm = os.getenv(parameter, eval(f"self.{parameter.lower()}"))
        except Exception as e:
            raise Exception(f"Error loading SQL server {parameter} : {e}")
        return prm

    @property
    def user(self) -> str:
        """
        User name of the SQL server.
        """
        try:
            user = os.getenv("DATABASE_USER", self._default_user)
        except Exception as e:
            raise Exception(f"Error loading SQL server user name: {e}")
        return user

    @property
    def pwd(self) -> str:
        """
        Password of the SQL server.
        """
        try:
            pwd = os.getenv("DATABASE_PWD", self._default_password)
        except Exception as e:
            raise Exception(f"Error loading SQL server password: {e}")

        return pwd

    def __repr__(self):
        """
        A full explicit representation of the connector for clarity
        """
        return (
            f"<{self.__class__.__name__}> (database_host={self.database_host}, user={self.user}, "
            f"database={self.database}, database_port={self.database_port}) "
            f"at {hex(id(self))}"
        )
