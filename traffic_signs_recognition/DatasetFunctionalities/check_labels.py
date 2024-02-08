"""
This file contains a script that allows the user to visualize the labels from the selected
folder. Labels are cropped from images and showed in batches of custom sizes.

Cool feature: if you display 10 or fewer images, you can press 1-9 to copy the picture's name to
clipboard. (0 for the 10th image)
"""

import os
import sys

import numpy as np

import cv2
import pandas as pd
import pyperclip

from traffic_signs_recognition.random_utils import (
    yolo_to_corner_values,
    read_class_names_from_yaml,
)


def display_multiple_pictures(current_pictures: dict, class_name: str):
    # Calculate canvas size and create it
    max_height = max(img.shape[0] for img in current_pictures.values()) + 20
    max_width = sum(img.shape[1] for img in current_pictures.values())
    canvas = np.ones((max_height, max_width, 3), dtype=np.uint8) * 255

    # Display images on the canvas
    current_width = 0
    print("_" * 10, class_name, "_" * 10)
    count = 1
    for image_name, image in current_pictures.items():
        h, w = image.shape[:2]
        canvas[:h, current_width : current_width + w] = image
        current_width += w

        cv2.putText(
            canvas,
            str(count) if count < 10 else "0",
            (current_width - int(w / 2), max_height - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
        count += 1

        # Print name in terminal just in case
        print(image_name)

    # Wait for user actions. Move forward or copy picture names for wrong labels.
    cv2.imshow(class_name, canvas)
    while True:
        key = cv2.waitKey(0)
        if chr(key).isnumeric() and int(chr(key)) - 1 < len(current_pictures.keys()):
            name_to_copy = list(current_pictures.keys())[int(chr(key)) - 1].split(".")[
                0
            ]
            pyperclip.copy(name_to_copy)
            print(f"Copied to clipboard: {name_to_copy}")
        if key == ord("w"):
            break
        if key == ord("m"):
            sys.exit()

    cv2.destroyAllWindows()


def extract_labels_for_index(
    path_to_dataset: str,
    df_label: pd.DataFrame,
    class_name: str,
    display_number: int = 10,
) -> None:
    # Iterate through the labels from the provided dataframe
    # Crop the labels and pass them to the display method 10 at a time
    current_pictures = {}
    for _, row in df_label.iterrows():
        current_image = os.path.join(path_to_dataset, row["split"], row["img_path"])

        x1, y1, x2, y2 = yolo_to_corner_values(
            [row["x1"], row["y1"], row["x2"], row["y2"]]
        )

        cropped_image = cv2.imread(current_image)[y1:y2, x1:x2]
        current_pictures[row["img_path"].split("\\")[-1]] = cropped_image

        # Display images 10 at a time. Overflow at the end just in case something weird happens.
        if len(current_pictures) >= display_number:
            display_multiple_pictures(current_pictures, class_name=class_name)
            current_pictures = {}

    if len(current_pictures) > 0:
        display_multiple_pictures(current_pictures, class_name=class_name)


def run_dataset_check(path_to_dataset: str):
    """Start the dataset if available at the provided path."""

    try:
        class_names = read_class_names_from_yaml(
            os.path.join(path_to_dataset, "data.yaml")
        )
    except FileNotFoundError:
        print("Cannot read the data.yaml file.")
        sys.exit()
    try:
        df = pd.read_csv(os.path.join(path_to_dataset, "dataset.csv"))
    except FileNotFoundError:
        print("Cannot read the dataset.csv file. Make sure to generate that first.")
        sys.exit()

    # Extract every occurrence for a label at a time and pass them to the extraction method
    for class_index, class_name in enumerate(class_names):
        df_label = df[df["label"] == class_index]
        extract_labels_for_index(path_to_dataset, df_label, class_name)


if __name__ == "__main__":
    PATH_TO_DATASET = "../datasets/original_dataset"
    run_dataset_check(PATH_TO_DATASET)
