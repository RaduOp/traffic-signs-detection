"""
This file contains a script that allows the user to visualize the labels from the selected
folder. Labels are cropped from images and showed in batches of custom sizes.

Cool feature: if you display 10 or fewer images you can press 1-9 to copy the picture's name to
clipboard. (0 for the 10th image)
"""

import argparse
import os
import time

import numpy as np

import cv2
import pandas as pd
import pyperclip

from traffic_signs_recognition.random_utils import (
    yolo_to_corner_values,
    read_class_names_from_yaml,
)


def display_multiple_pictures(
    img_list: list[np.ndarray], image_names: list[str], class_name: str
):
    max_height = max(img.shape[0] for img in img_list)
    max_width = sum(img.shape[1] for img in img_list)

    canvas = np.ones((max_height, max_width, 3), dtype=np.uint8) * 255

    current_width = 0
    for img in img_list:
        h, w = img.shape[:2]
        canvas[:h, current_width : current_width + w] = img
        current_width += w

    cv2.imshow(class_name, canvas)
    print("_" * 10, class_name, "_" * 10)
    if len(image_names) > 10:
        for name in image_names:
            print(name)
    while True:
        key = cv2.waitKey(0)
        if chr(key).isnumeric() and int(chr(key)) - 1 < len(image_names):
            pyperclip.copy(image_names[int(chr(key)) - 1])
            print("Copied to clipboard: ", image_names[int(chr(key)) - 1])
        if key == ord("w"):
            break
        if key == ord("m"):
            quit()

    cv2.destroyAllWindows()


def check_labels_from_dataset(path_to_dataset: str, display_number: int = 10) -> None:
    class_names = read_class_names_from_yaml(os.path.join(path_to_dataset, "data.yaml"))
    df = pd.read_csv(os.path.join(path_to_dataset, "dataset.csv"))
    for class_index in range(len(class_names)):
        current_pictures = []
        current_pictures_names = []
        df_label = df[df["label"] == class_index]
        for _, row in df_label.iterrows():
            current_image = os.path.join(path_to_dataset, row["split"], row["img_path"])

            x1, y1, x2, y2 = yolo_to_corner_values(
                [row["x1"], row["y1"], row["x2"], row["y2"]]
            )
            cropped_image = cv2.imread(current_image)[y1:y2, x1:x2]
            current_pictures.append(cropped_image)
            current_pictures_names.append(row["img_path"].split("\\")[-1])

            if len(current_pictures) >= display_number:
                display_multiple_pictures(
                    current_pictures,
                    current_pictures_names,
                    class_name=class_names[class_index],
                )
                current_pictures = []
                current_pictures_names = []

        # Overflow, just in case
        if len(current_pictures) > 0:
            display_multiple_pictures(
                current_pictures,
                current_pictures_names,
                class_name=class_names[class_index],
            )


if __name__ == "__main__":
    path_to_dataset = "../datasets/original_dataset"
    display_number = 9
    check_labels_from_dataset(path_to_dataset)
