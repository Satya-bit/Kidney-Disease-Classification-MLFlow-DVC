#To read and return the config.yaml and params.yaml
# This would remioan constant we won't be changing it. So we write it in constants

from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")  #Path to avoid Windows path error
PARAMS_FILE_PATH = Path("params.yaml")