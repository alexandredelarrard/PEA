from omegaconf import DictConfig, OmegaConf
import glob
from pathlib import Path


def read_config(path: str) -> DictConfig:
    """Read configuration from YAML files"""
    config_path = Path(path)
    main_config_path = config_path / "configs.yml"

    if not main_config_path.exists():
        raise FileNotFoundError(f"Main config file not found: {main_config_path}")

    config = OmegaConf.load(main_config_path)

    # Load other config files
    for config_file in config_path.glob("**/*.yml"):
        if config_file.name != "configs.yml":
            sub_config = OmegaConf.load(config_file)
            config = OmegaConf.merge(config, sub_config)

    return config
