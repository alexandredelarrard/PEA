#
# Copyright (c) 2022 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-
import logging

from box import Box
from forecast_fl.postgres_db.server import SQLServer
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


def load_database() -> SQLServer:
    """Sets up SQL server connection"""
    # Create DB connector
    database = SQLServer()
    database.load_config()

    # Create DB
    with database as _:
        logging.info("DATA: Created database")
        Base.metadata.create_all(database.engine)

    return database
