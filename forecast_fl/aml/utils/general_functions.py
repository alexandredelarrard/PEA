#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

from os import environ
from typing import Union

from azureml.core import Datastore, Environment, Workspace
from azureml.core.conda_dependencies import CondaDependencies
from azureml.data.abstract_datastore import AbstractDatastore
from azureml.exceptions import UserErrorException


def get_workspace() -> Workspace:

    workspace = Workspace(
        workspace_name=environ["WORKSPACE_NAME"],
        _location=environ["LOCATION"],
        subscription_id=environ["SUBSCRIPTION_ID"],
        resource_group=environ["RESOURCE_GROUP_NAME"],
    )
    return workspace


def get_blob_storage(workspace: Workspace, DATA_INPUTS, DATA_INPUTS_FL360) -> Union[Datastore, AbstractDatastore]:

    try:
        blob_datastore = Datastore(workspace, name=environ["DATASTORE"])

        # upload inputs
        blob_datastore.upload(
            src_dir="./local_data/",
            target_path=DATA_INPUTS + "/mapping_files",
            overwrite=True,
        )

        # upload inputs
        blob_datastore.upload(
            src_dir="./local_data_application/",
            target_path=DATA_INPUTS_FL360,
            overwrite=True,
        )

    except UserErrorException:
        blob_datastore = Datastore.register_azure_blob_container(
            workspace=workspace,
            datastore_name=environ["DATASTORE"],
            container_name=environ["CONTAINER_NAME"],
            account_name=environ["STORAGE_ACCOUNT"],
            account_key=environ["adlKey"],
        )

        # upload inputs
        blob_datastore.upload(
            src_dir="./local_data/",
            target_path=DATA_INPUTS + "/mapping_files",
            overwrite=True,
        )

        # upload inputs
        blob_datastore.upload(
            src_dir="./local_data_application/",
            target_path=DATA_INPUTS_FL360,
            overwrite=True,
        )

    return blob_datastore


def define_environment_azure_ml() -> Environment:

    # Specify docker steps as a string, to manually install broken deps.
    conda = CondaDependencies(conda_dependencies_file_path="environment.yml")
    env = Environment(name="outil_prevision_env")
    dockerfile = rf"""
    FROM mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04

    # FILES COPY
    #RUN mkdir /data\
    #RUN mkdir /data/inputs/

    # UPDATES CONDA
    # RUN conda update -n base -c defaults conda

    # ENV VAR
    ENV DATABASE_PWD="{environ['DATABASE_PWD']}"
    ENV DATABASE_USER="{environ['DATABASE_USER']}"
    ENV DATABASE={environ['DATABASE']}
    ENV DATABASE_HOST={environ['DATABASE_HOST']}
    ENV PORT={environ['DATABASE_PORT']}
    """
    # Fichier YAML d'environnement dans le dossier du script

    # Set base image to None, because the image is defined by dockerfile.
    env.docker.base_image = None
    env.docker.base_dockerfile = dockerfile
    env.python.conda_dependencies = conda

    return env
