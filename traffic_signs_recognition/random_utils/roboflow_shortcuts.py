"""
Some shortcuts for various Roboflow actions.
LOAD .env BEFORE USING THESE METHODS!!
"""

import os

from roboflow import Roboflow


def setup_roboflow_project():
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace(os.getenv("ROBOFLOW_WORKSPACE")).project(
        os.getenv("ROBOFLOW_PROJECT")
    )

    return project


def download_dataset(dataset_version: int, download_path: str) -> None:
    """Download a dataset from Roboflow."""
    project = setup_roboflow_project()
    project.version(dataset_version).download(
        os.getenv("ROBOFLOW_FORMAT"), location=download_path
    )


def deploy_model(
    dataset_version: int,
    checkpoint_path: str,
    model_type: str = "traffic_signs_recognition",
) -> None:
    """Deploy a model to Roboflow."""
    project = setup_roboflow_project()
    project.version(dataset_version).deploy(model_type, checkpoint_path)


def upload_data(path_to_folder: str, path_to_yaml_file: str) -> None:
    """Upload images to roboflow via API. Can't upload label map for images with label.
    TODO: WILL NEED A WORKAROUND
    """
    project = setup_roboflow_project()

    # project.upload(path_to_yaml_file)
    for image_file, txt_file in zip(
        os.listdir(os.path.join(path_to_folder, "images")),
        os.listdir(os.path.join(path_to_folder, "labels")),
    ):
        project.upload(
            image_path=os.path.join(path_to_folder, "images", image_file),
            annotation_path=os.path.join(path_to_folder, "labels", txt_file),
            split="train",
            batch_name="ADDED_VIA_API",
            is_prediction=True,
        )
