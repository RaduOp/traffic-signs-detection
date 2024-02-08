"""
From a folder with images and labels create a csv file for benchmarking. Each label will need to
be provided with a difficulty.
"""

import os
import sys

import cv2
import pandas as pd

from traffic_signs_recognition.random_utils import (
    draw_custom_rectangle,
    yolo_to_corner_values,
)


def create_benchmark_csv(folder_path: str):
    """Create benchmark.csv file from a folder with images and labels."""

    all_images = [file for file in os.listdir(folder_path) if file.endswith(".jpg")]
    all_data = []

    for image_name in all_images:
        original_image = cv2.imread(os.path.join(folder_path, image_name))

        with open(
            os.path.join(folder_path, image_name[:-3] + "txt"), "r", encoding="utf-8"
        ) as f:
            for line in f:
                label = [float(el) for el in line.strip().split(" ")]
                current_image = original_image.copy()
                drawn_image = draw_custom_rectangle(
                    current_image,
                    yolo_to_corner_values(
                        label[1:],
                        image_height=original_image.shape[0],
                        image_width=original_image.shape[1],
                    ),
                )

                cv2.imshow("test", drawn_image)
                while True:
                    key = cv2.waitKey(0)
                    if chr(key).isnumeric() and int(chr(key)) - 1 < 3:
                        print(f"Label difficulty set to: {chr(key)}")
                        break

                    if key == ord("q"):
                        sys.exit()
                    else:
                        print("Please select a value from 1 to 3.")

                current_df_line = [
                    int(label[0]),
                    *yolo_to_corner_values(
                        label[1:],
                        image_height=original_image.shape[0],
                        image_width=original_image.shape[1],
                    ),
                    int(chr(key)),
                    image_name.split("/")[-1],
                ]

                all_data.append(current_df_line)
    result_df = pd.DataFrame(
        all_data,
        columns=["label", "x1", "y1", "x2", "y2", "difficulty", "img_name"],
    )
    result_df.to_csv(os.path.join(folder_path, "benchmark.csv"), index=False)
    return all_data


print(create_benchmark_csv("test_data"))
cv2.destroyAllWindows()
# model_ckpt_path = "../YOLOV8/runs/detect/train8/weights/best.pt"
# image_detector = YoloDetector(model_ckpt_path, 1920)
