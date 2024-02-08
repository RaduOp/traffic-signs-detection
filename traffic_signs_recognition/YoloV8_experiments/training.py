"""
Download dataset and modify the yaml file accordingly.
Train a YOLOv8 or YOLOv5 model.
"""

import os
import shutil
import sys

from ultralytics import YOLO
import yaml
import dotenv
from traffic_signs_recognition.random_utils.roboflow_shortcuts import download_dataset


dotenv.load_dotenv(dotenv.find_dotenv())


def manage_download(dataset_version: int, download_path: str):
    if os.path.isdir(download_path):
        shutil.rmtree(download_path)
        os.makedirs(download_path)
    else:
        os.makedirs(download_path)

    download_dataset(dataset_version, download_path)


def start_training_yolov8(path_to_dataset: str, pretrained_model_path: str):
    def update_yaml_file():
        train_path = os.path.join("train")
        valid_path = os.path.join("valid")
        with open(path_to_dataset + "/data.yaml", "r", encoding="utf-8") as file:
            data: dict = yaml.safe_load(file)

        # Update the specified field
        data["train"] = train_path
        data["val"] = valid_path
        if "test" in data.keys():
            del data["test"]

        with open(path_to_dataset + "/data.yaml", "w", encoding="utf-8") as file:
            yaml.dump(data, file)

    update_yaml_file()

    model = YOLO(pretrained_model_path)
    # Set up the training configuration
    config = {
        "epochs": 100,
        "pretrained": True,
        "imgsz": 640,
        "cache": True,
        "workers": 0,
        "batch": 32,
        "device": 0,
        "val": True,
        "weight_decay": 0.0001,
        "data": os.path.abspath(os.path.join(path_to_dataset, "data.yaml")),
    }

    model.train(**config)


# git clone https://github.com/ultralytics/yolov5 - to work
def start_training_yolov5(path_to_dataset: str, pretrained_model_path: str):
    try:
        from yolov5 import train
    except:
        print(
            "You need to clone the yolov5 repo first.\n"
            "git clone https://github.com/ultralytics/yolov5)"
        )
        sys.exit()

    def update_yaml_file():
        original_yaml = os.path.join(path_to_dataset, "data.yaml")
        copy_yaml = os.path.join("yolov5", "data.yaml")
        shutil.copy2(original_yaml, copy_yaml)

        train_path = "../../datasets/original_dataset/train"
        valid_path = "../../datasets/original_dataset/valid"
        with open(copy_yaml, "r", encoding="utf-8") as file:
            data: dict = yaml.safe_load(file)

        # Update the specified field
        data["train"] = train_path
        data["val"] = valid_path
        if "test" in data.keys():
            del data["test"]

        with open(copy_yaml, "w", encoding="utf-8") as file:
            yaml.dump(data, file)

    update_yaml_file()

    # Set up the training configuration
    config = {
        "weights": pretrained_model_path,
        "epochs": 100,
        "imgsz": 640,
        "cache": True,
        "workers": 0,
        "batch": 16,
        "device": 0,
        "val": True,
        "data": "yolov5/data.yaml",
    }

    train.run(**config)


if __name__ == "__main__":
    PATH_TO_DATASET = "../datasets/original_dataset"
    PRETRAINED_MODEL_PATH = "../pretrained_models/yolov8s.pt"

    # manage_download(10, download_path=PATH_TO_DATASET)
    start_training_yolov8(PATH_TO_DATASET, PRETRAINED_MODEL_PATH)
