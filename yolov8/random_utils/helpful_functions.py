"""
This file contains a collection of functions that do small repetitive things.
"""

import yaml


def read_class_names_from_yaml(path_to_yaml_file: str) -> list[str]:
    """
    Reads the yaml file for the current dataset. It's great for dataset manipulation or
    visualization of class name instead of indexes.

    :param path_to_yaml_file: location of the data.yaml file, usually in the dataset folder
    :return: a list of class names
    """
    with open(path_to_yaml_file, "r") as f:
        return yaml.safe_load(f)['names']
