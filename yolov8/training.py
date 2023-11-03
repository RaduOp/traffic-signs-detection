import os
import shutil
from ultralytics import YOLO
import yaml
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv())


def download_dataset(dataset_version):
    if os.path.isdir("datasets"):
        shutil.rmtree("datasets")
        os.mkdir("datasets")
    else:
        os.mkdir("datasets")
    from roboflow import Roboflow

    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace(os.getenv("ROBOFLOW_WORKSPACE")).project(os.getenv("ROBOFLOW_PROJECT"))
    dataset = project.version(dataset_version).download(os.getenv(
        "ROBOFLOW_FORMAT"), location="datasets")


def start_training():
    model = YOLO('pretrained_models/yolov8n.pt')
    results = model.train(data='datasets/data.yaml', epochs=50, imgsz=640, cache=True,
                          workers=0)


if __name__ == '__main__':
    download_dataset(4)
    # start_training()
