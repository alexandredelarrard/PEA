from dataclasses import dataclass
from logging import Logger, getLogger
from os import getenv
from typing import List, Union

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import KeyVaultSecret, SecretClient
from azureml.core import Run
from azureml.core.run import _OfflineRun, _SubmittedRun
from dotenv import load_dotenv


class OfflineException(Exception):
    pass


class EnvironmentVariableMissing(EnvironmentError):
    pass


class SecretException(Exception):
    pass


def get_run_id() -> str:
    """
    Returns: run id (str) if a pipeline job is running
    """

    run = Run.get_context(allow_offline=True)
    if not isinstance(run, _OfflineRun):
        if run.parent is not None:
            return str(run.parent.id)
    return ""


@dataclass
class CloudStatus:
    @property
    def run_instance(self):
        return Run.get_context(allow_offline=True)

    @property
    def offline_run(self) -> bool:
        return isinstance(Run.get_context(allow_offline=True), _OfflineRun)

    @property
    def offline(self) -> bool:
        if self.offline_run:
            return True
        return False

    @property
    def aml_pipeline_run(self) -> bool:
        if isinstance(Run.get_context(allow_offline=True), _SubmittedRun):
            return True
        return False


@dataclass
class Secrets:
    status: CloudStatus = CloudStatus()
    key_vault_name: str = "TEST_KEY_VAULT"
    logger: Logger = getLogger(__name__)

    @staticmethod
    def keys() -> List[str]:
        return [
            "BASE_PATH",
            "DATABASE_USER",
            "DATABASE_PWD",
            "DATABASE",
            "DATABASE_HOST",
            "DATABASE_PORT",
            "WORKSPACE_NAME",
            "SUBSCRIPTION_ID",
            "RESOURCE_GROUP_NAME",
            "STORAGE_ACCOUNT",
            "LOCATION",
            "DATASTORE",
            "DATA_INPUTS",
            "PIPELINE_OUTPUTS",
            "AML_CPU_CLUSTER_NAME",
            "AML_ALLOW_REUSE",
        ]

    @property
    def resource_group(self) -> str:
        value = getenv("RESOURCE_GROUP_NAME")
        if value is None:
            raise EnvironmentVariableMissing(f"Cannot find Environment Variabable RESOURCE_GROUP_NAME")
        return value

    @property
    def key_vault_uri(self) -> str:
        return f"{self.key_vault_name}_{self.resource_group}"

    @property
    def credential(self) -> DefaultAzureCredential:
        if self.status.offline:
            raise OfflineException("Cannot create AzureCredentials with status offline")
        return DefaultAzureCredential()

    @property
    def client(self) -> SecretClient:
        if self.status.offline:
            raise OfflineException("Cannot create SecretClient with status offline")
        return SecretClient(vault_url=self.key_vault_uri, credential=self.credential)

    def upload_secret_from_dotenv(self, name: str) -> KeyVaultSecret:
        value = getenv(name)
        if value is None:
            raise EnvironmentVariableMissing(f"Cannot find Environment Variable {name} for upload")
        secret = self.client.set_secret(name, value)
        if secret is None:
            raise SecretException("Failed to Retrieve Secret Value")
        return secret

    def upload_dotenv(self, path: str = ".env") -> None:
        if self.status.offline:
            raise OfflineException("Cannot upload dotenv with status offline")
        load_dotenv(path)
        for key in self.keys():
            _ = self.upload_secret_from_dotenv(key)

    def get_secret(self, name: str) -> str:
        if self.status.offline:
            raise OfflineException("Cannot get secret with status offline")
        secret_value = self.client.get_secret(name).value
        if secret_value is None:
            raise SecretException("Failed to Retrieve Secret Value")
        return secret_value

    def set_secret(self, name: str, value: str) -> KeyVaultSecret:
        if self.status.offline:
            raise OfflineException("Cannot set secret with status offline")
        secret = self.client.set_secret(name, value)
        if secret is None:
            raise SecretException("Failed to Retrieve Secret Value")
        return secret
