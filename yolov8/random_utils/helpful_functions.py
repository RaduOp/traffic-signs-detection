import yaml


def read_class_names_from_yaml(path_to_yaml_file):
    with open(path_to_yaml_file, "r") as f:
        class_names: list = yaml.safe_load(f)['names']
    return class_names
