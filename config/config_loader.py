import yaml
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def load_config():
    """
    Loads configuration from a YAML file.

    This function reads a YAML configuration file located at a predefined
    path (CONFIG_PATH) and parses its content into a Python dictionary
    using a safe YAML loader. The returned dictionary can then be used
    to access configuration settings.

    :return: A dictionary containing the parsed configuration from the YAML file.
    :rtype: dict
    """
    with open(CONFIG_PATH, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg

CONFIG = load_config()