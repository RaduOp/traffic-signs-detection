"""
This file contains a simple script that cycles through images and labels from a folder. It will
display the labels on the image and if you press "d" (you can remap this) it will delete the image.

It's great for images collected automatically by the model as they are not very accurate at the
moment.

It also helps avoid the slow process of deleting an image that's been uploaded to the dataset on
roboflow. It takes roughly 10-20 seconds to delete an image on Roboflow.
"""
import argparse
import os

import cv2
from yolov8.random_utils.label_transposing import yolo_to_corner_values


def check_collected_images(path_to_folder: str, delete_key: str):
    for image_name in os.listdir(os.path.join(path_to_folder, "images")):
        current_picture = os.path.join(path_to_folder, "images", image_name)
        current_label = os.path.join(path_to_folder, "images", image_name[:-4] + ".txt")

        image = cv2.imread(current_picture)

        with open(current_label, 'r') as label_file:
            for line in label_file:
                label = [float(el) for el in line.strip().split(" ")]
                x1, y1, x2, y2 = map(int, yolo_to_corner_values(label[1:]))
                image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

        cv2.imshow("img", image)
        key = cv2.waitKey(0)
        if key == ord(delete_key):
            if os.path.exists(current_label) and os.path.exists(current_picture):
                os.remove(current_picture)
                os.remove(current_label)
                print(f"Image and label deleted: ", image_name, image_name[:-4] + ".txt")
            else:
                print(f"Delete failed, label or image is missing: ", image_name, image_name[:-4] +
                      ".txt")

        else:
            print("Image approved: ", image_name)


def parse_arguments():
    parser = argparse.ArgumentParser(description="A script that allows you to quickly iterate "
                                                 "trough a bunch of images with labels and decide "
                                                 "which ones you want to keep.")
    parser.add_argument(
        "--delete_key",
        required=False,
        type=str,
        default="d",
        help="Change the default delete key 'd'."
    )
    parser.add_argument(
        "--location",
        type=str,
        required=True,
        help="Path to folder with images and labels.",

    )

    return parser.parse_args()


if __name__ == "__main__":
    user_args = parse_arguments()
    check_collected_images(user_args.location, user_args.delete_key)
