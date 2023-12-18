import os
import shutil

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


def update_yaml_file(path_to_dataset):
    train_path = os.path.join("train")
    valid_path = os.path.join("valid")
    with open(path_to_dataset + "/data.yaml", "r") as file:
        data: dict = yaml.safe_load(file)

    # Update the specified field
    data["train"] = train_path
    data["val"] = valid_path
    if "test" in data.keys():
        del data["test"]

    with open(path_to_dataset + "/data.yaml", "w") as file:
        yaml.dump(data, file)


def start_training(path_to_dataset: str, pretrained_model_path: str):
    model = YOLO(pretrained_model_path)
    # Set up the training configuration
    config = {
        "epochs": 20,
        "imgsz": 640,
        "cache": True,
        "workers": 0,
        "batch": -1,
        "device": 0,
        "val": True,
        "data": os.path.join(path_to_dataset, "data.yaml"),
    }

    model.train(**config)


if __name__ == "__main__":
    path_to_dataset = "../datasets/original_dataset"
    pretrained_model_path = "../runs/detect/train16/weights/best.pt"

    manage_download(10, download_path=path_to_dataset)
    update_yaml_file(path_to_dataset)
    start_training(path_to_dataset, pretrained_model_path)
