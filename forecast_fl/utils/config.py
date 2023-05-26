"""
Config manager
"""

import getpass
import logging
import os
from typing import Type, Union

import yaml
from _io import TextIOWrapper
from box import Box


def _user_specific_file(filename: str) -> Union[None, str]:
    """Find user specific files for a filename.
    E.g. user_specific_file(config.yml) = config.$USER.yml if the file
    exists, else returns None
    """
    username = getpass.getuser().lower().replace(" ", "_")
    filepath, file_extension = os.path.splitext(filename)
    user_filename = filepath + "." + username + file_extension
    if os.path.isfile(user_filename) and os.access(user_filename, os.R_OK):
        return user_filename
    else:
        return None


def _read(filename: str, loader: Type[yaml.FullLoader]) -> Box:
    """Read any yaml file as a Box object"""

    if os.path.isfile(filename) and os.access(filename, os.R_OK):
        with open(filename, "r", encoding="utf8") as f:
            try:
                config_dict = yaml.load(f, Loader=loader)
            except yaml.YAMLError as exc:
                logging.info(exc)
        return Box(config_dict)
    else:
        raise FileNotFoundError(filename)


def _overwrite_with_user_specific_file(config: Box, filename: str) -> Box:
    """Overwrite the config files with user specific files"""
    user_filename = _user_specific_file(filename)
    if user_filename:
        logging.info(f"{filename} overwritten by {user_filename}")
        user_config: Box = _read(user_filename, loader=CustomYamlLoader)
        config.merge_update(user_config)

    return config


class CustomYamlLoader(yaml.FullLoader):
    """Add a custom constructor "!include" to the YAML loader.
    "!include" allows to read parameters in another YAML file as if it was
    the main one.
    Examples:
        To read the parameters listed in credentials.yml and assign them to
        credentials in logging.yml:
        ``credentials: !include credentials.yml``
        To call: config.credentials
    """

    def __init__(self, stream: TextIOWrapper):
        self._root = os.path.split(stream.name)[0]
        super(CustomYamlLoader, self).__init__(stream)

    def include(self, node: yaml.nodes.ScalarNode) -> Box:
        """Read yaml files as Box objects and overwrite user specific files
        Example: !include model.yml, will be overwritten by model.$USER.yml
        """

        filename: str = os.path.join(self._root, self.construct_scalar(node))  # type: ignore
        subconfig: Box = _read(filename, loader=CustomYamlLoader)
        subconfig = _overwrite_with_user_specific_file(subconfig, filename=filename)

        return subconfig


CustomYamlLoader.add_constructor("!include", CustomYamlLoader.include)  # type: ignore


def get_filepath(config: Box, folder_key: str, file_key: str) -> str:
    base_directory = config.paths.base_directory
    folder = config.paths.get(folder_key).folder
    filename = config.paths.get(folder_key).paths.get(file_key)
    return os.path.join(base_directory, folder, filename)


def get_rdb_url(config: Box) -> str:
    dialect = config.database_connection.rdbms.dialect
    user_name = config.database_connection.rdbms.config.user
    user_password = config.database_connection.rdbms.config.password
    host = config.database_connection.rdbms.config.host
    port = config.database_connection.rdbms.config.port
    db_name = config.database_connection.rdbms.config.dbname
    rdb_url = f"{dialect}://{user_name}:{user_password}@{host}:{port}/{db_name}"
    return rdb_url


class Config:
    def __init__(self, config_path: str):
        self.config = None
        self._config_path = config_path

    def read(self) -> Box:
        """Reads main config file"""
        if os.path.isfile(self._config_path) and os.access(self._config_path, os.R_OK):
            config = _read(filename=self._config_path, loader=CustomYamlLoader)
            self.config = _overwrite_with_user_specific_file(config, filename=self._config_path)
            return self.config
        else:
            raise FileNotFoundError(self._config_path)
