import os
import shutil

from traffic_signs_recognition.random_utils.roboflow_shortcuts import download_dataset
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv())


def manage_download(dataset_version: int, download_path: str):
    if os.path.isdir(download_path):
        shutil.rmtree(download_path)
        os.makedirs(download_path)
    else:
        os.makedirs(download_path)

    download_dataset(dataset_version, download_path)


def start_training():
    pass


if __name__ == "__main__":
    path_to_dataset = "../datasets/original_dataset"

    manage_download(10, download_path=path_to_dataset)

    # start_training()
