"""
This file contains a script that allows the user to visualize the labels from the selected
folder. Labels are cropped from images and showed in batches of custom sizes.

Cool feature: if you display 10 or fewer images you can press 1-9 to copy the picture's name to
clipboard. (0 for the 10th image)

NOTE: Cycling through all txt files for each label can be inefficient so a better algorithm might
be needed in the future.
As of right now there are 2k pictures in the dataset, and it takes less than 0.1ms to open every
txt file and read the labels. Sounds like a future me problem :)
"""

import argparse
import os

import numpy as np

import cv2
import pyperclip

from yolov8.random_utils.helpful_functions import read_class_names_from_yaml
from yolov8.random_utils.label_transposing import yolo_to_corner_values


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


def check_labels_from_dataset(
    path_to_folder: str, path_to_data_file, display_number: int = 100
) -> None:
    class_names = read_class_names_from_yaml(path_to_data_file)

    for index in range(len(class_names)):
        current_pictures = []
        current_pictures_names = []

        for image_name in os.listdir(os.path.join(path_to_folder, "images")):
            current_image = os.path.join(path_to_folder, "images", image_name)
            current_label = os.path.join(
                path_to_folder, "labels", image_name[:-4] + ".txt"
            )

            with open(current_label, "r") as label_file:
                for line in label_file:
                    label = [float(el) for el in line.strip().split(" ")]
                    if int(label[0]) == index:
                        x1, y1, x2, y2 = yolo_to_corner_values(label[1:])
                        cropped_image = cv2.imread(current_image)[y1:y2, x1:x2]
                        current_pictures.append(cropped_image)
                        current_pictures_names.append(image_name)

            if len(current_pictures) >= display_number:
                display_multiple_pictures(
                    current_pictures,
                    current_pictures_names,
                    class_name=class_names[index],
                )
                current_pictures = []
                current_pictures_names = []

        if len(current_pictures) > 0:
            display_multiple_pictures(
                current_pictures, current_pictures_names, class_name=class_names[index]
            )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="A script that allows you to crop labels from "
        "images and display them a few at time to check "
        "for mislabeled objects."
    )
    parser.add_argument(
        "--display_number",
        required=False,
        type=int,
        default=5,
        help="Change the amount of pictures to be displayed at once.",
    )
    parser.add_argument(
        "--location",
        type=str,
        required=True,
        help="Path to folder with images and labels.",
    )

    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to yaml file that contains class_names.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    user_args = parse_arguments()
    print(
        "Info:\nPress 'w' for next batch.\n"
        "Press 'm' to close.\n"
        "Press '1' to '0' to copy image name to clipboard.\n"
    )
    check_labels_from_dataset(
        user_args.location, user_args.data_file, user_args.display_number
    )
