import os

import cv2
import numpy as np

from yolov8.random_utils.label_transposing import corner_values_to_yolo


def save_one(
    frame: np.ndarray,
    coordinates: list[dict],
    save_location: str,
    file_name: str,
    image_extension: str = ".png",
):
    """
    Save the cropped image and it's labels.
    Method asumes that pictures will be saved under "images" sub folder and txt files under
    "labels" sub folder.

    WILL CREATE FOLDERS IF NOT FOUND.
    WILL NOT OVERRIDE IMAGES.
    """
    if not os.path.exists(os.path.join(save_location, "images")):
        os.mkdir(os.path.join(save_location, "images"))
    if not os.path.exists(os.path.join(save_location, "labels")):
        os.mkdir(os.path.join(save_location, "labels"))

    index = 0
    while True:
        img_path = os.path.join(
            save_location, "images", file_name + f"_{index}" + image_extension
        )
        txt_path = os.path.join(
            save_location, "labels", file_name + f"_{index}" + ".txt"
        )
        if not os.path.exists(img_path) and not os.path.exists(txt_path):
            break
        index += 1

    # Save image
    cv2.imwrite(img_path, frame)

    # Save labels to txt
    with open(txt_path, "a") as file:
        for detection in coordinates:
            x_center, y_center, width, height = corner_values_to_yolo(
                detection["coords"]
            )
            file.write(
                f"{int(detection['class_index'])} {x_center} {y_center} {width}"
                f" {height}\n"
            )
