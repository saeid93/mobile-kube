from .constants import CONFIGS_PATH
import os
import json


def config_reader(mode, config_folder, type_of):
    """
    read the config files
    """
    if type_of == 'check':
        file = "config_check_env.json"
    elif type_of == 'learn':
        file = "config_run.json"
    else:
        ValueError("unknown config type")
    config_file_path = os.path.join(CONFIGS_PATH, mode, config_folder,
                                    file)

    with open(config_file_path) as cf:
        config = json.loads(cf.read())

    return config, config_file_path
