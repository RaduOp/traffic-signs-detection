"""
This file is meant for collecting data from using the model from a video.
It's still in testing and a lot will change.

The final version will most likely be a script.
"""
import os.path
import time

import cv2
import argparse
from yolov8.dataset_handling.database_health_check import dataset_health_check
from yolov8.CroppingLabels.get_random_crops_from_frame import ImageCropper
from yolov8.detectors import YoloDetector
from yolov8.random_utils.helpful_functions import read_class_names_from_yaml
from yolov8.CroppingLabels.save_crops import save_one
from yolov8.random_utils.draw_on_image import *


def process_video(image_cropper: ImageCropper, image_detector: YoloDetector,
                  path_to_video: str, save_location: str, skip_nr: int = 3,
                  show_output: bool = True) -> None:
    cap = cv2.VideoCapture(path_to_video)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    if show_output:
        class_names = image_detector.get_class_names()
    skip = -1
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            skip += 1
            if skip < skip_nr:
                continue
            boxes = image_detector.get_results_as_list(frame)
            for crop in image_cropper.get_crops(frame, boxes):
                save_one(crop[0], crop[1], save_location, "auto_collected")
            if show_output:
                for box in boxes:
                    if box["conf"] > 0.5:
                        draw_custom_rectangle(frame, box["coords"])
                        draw_rectangle_name(frame, box["coords"], class_names[box["class_index"]],
                                            box['conf'])
                cv2.imshow('Processed Frame', frame)
            skip = -1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


# def parse_arguments():
#     parser = argparse.ArgumentParser(description="Testing the model on video or live on main "
#                                                  "screen")
#
#     parser.add_argument("--video", action="store_true",
#                         help="Enable to save the processed footage.")
#     parser.add_argument("--save_location", action="store_true",
#                         help="Enable to save the processed footage.")
#     parser.add_argument("--skip_frames", action="store_true",
#                         help="Enable to save the processed footage.")
#     parser.add_argument("--show_output", action="store_true",
#                         help="Enable to save the processed footage.")
#     parser.add_argument("--model_checkpoint", action="store_true",
#                         help="Enable to save the processed footage.")
#
#     user_args = parser.parse_args()
#     return user_args


if __name__ == '__main__':
    # user_args = parse_arguments()
    model_ckpt_path = '../runs/detect/train13/weights/best.pt'

    number_of_occurrences = dataset_health_check("../datasets/train")
    whitelisted_classes = [index for index, item in enumerate(number_of_occurrences) if
                           item < 50]

    base = "../collected_images/"
    index = 0
    while True:
        if not os.path.exists(os.path.join(base, f"run_{index}")):
            os.mkdir(os.path.join(base, f"run_{index}"))
            os.mkdir(os.path.join(base, f"run_{index}", "labels"))
            os.mkdir(os.path.join(base, f"run_{index}", "images"))
            path_to_save = os.path.join(base, f"run_{index}")
            break
        index += 1

    image_detector = YoloDetector(model_ckpt_path, 1920)
    image_cropper = ImageCropper(whitelisted_classes)

    process_video(image_cropper, image_detector,
                  "../rendered.mp4", skip_nr=3, save_location=path_to_save)
