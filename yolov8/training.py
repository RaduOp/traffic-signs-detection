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


def update_yaml_file(path_to_folder):
    train_path = os.path.join("train")
    valid_path = os.path.join("valid")
    with open(path_to_folder + "/data.yaml", 'r') as file:
        data: dict = yaml.safe_load(file)

    # Update the specified field
    data["train"] = train_path
    data["val"] = valid_path
    del data["test"]

    with open(path_to_folder + "/data.yaml", 'w') as file:
        yaml.dump(data, file)


def start_training():
    model = YOLO('pretrained_models/yolov8n.pt')
    # model = model.to("cuda")
    results = model.train(data='datasets/data.yaml', epochs=100, imgsz=640, cache="ram",
                          workers=0, batch=-1, device=0, val=True)


if __name__ == '__main__':
    download_dataset(8)
    update_yaml_file("datasets")
    start_training()
