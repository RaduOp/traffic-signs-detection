"""
This file is meant for live testing of the trained model.
"""
import os.path
import time

import cv2
from mss import mss
import numpy as np
import argparse
from database_health_check import dataset_health_check
from process_detections import ProcessImage
from detectors import YoloDetector
from yolov8.random_utils.helpful_functions import read_class_names_from_yaml


def test_on_video(image_processing: ProcessImage, image_detector: YoloDetector,
                  path_to_video: str) -> \
        None:
    cap = cv2.VideoCapture(path_to_video)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    skip = -1
    while cap.isOpened():
        start = time.time()
        ret, frame = cap.read()
        if ret:
            # frame = np.array(frame)
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            skip += 1
            if skip < 1:
                continue
            boxes = image_detector.get_raw_detection_results(frame)
            frame = image_processing.process_frame(frame, boxes)
            cv2.imshow('Processed Frame', frame)
            print("Time per frame: ", (time.time() - start) / 3)
            skip = -1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def test_live_on_screen_with_multiple_monitors(image_processing: ProcessImage, image_detector:
YoloDetector) -> None:
    bounding_box = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
    sct = mss()
    monitor_number = 1
    monitor = {
        "top": 0,
        "left": 0,
        "width": 1920,
        "height": 1080,
        "mon": monitor_number,
    }

    # Grab the data
    while True:
        start = time.time()
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        boxes = image_detector.get_raw_detection_results(frame)
        frame = image_processing.process_frame(frame, boxes)
        # cv2.imshow('Processed Frame', frame)
        print("Time per frame: ", time.time() - start)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def parse_arguments():
    parser = argparse.ArgumentParser(description="Testing the model on video or live on main "
                                                 "screen")
    parser.add_argument(
        "--method",
        choices=["screen", "video"],
        default="screen",
        help="Choose the testing method. 'screen' for live feed of the main screen, 'video' for "
             "testing on a prerecorded video."
    )

    parser.add_argument("--save", action="store_true",
                        help="Enable to save the processed footage.")

    user_args = parser.parse_args()
    return user_args


if __name__ == '__main__':
    user_args = parse_arguments()
    dataset_yaml_path = "datasets/data.yaml"
    model_ckpt_path = 'runs/detect/train3/weights/best.pt'
    class_names = read_class_names_from_yaml("datasets/data.yaml")

    number_of_occurrences = dataset_health_check("datasets/train")
    print(number_of_occurrences)
    whitelisted_classes = [index for index, item in enumerate(number_of_occurrences) if
                           item < 100]

    for name, number in zip(class_names, number_of_occurrences):
        print(name, ": ", number)

    path_to_save = ""
    save_crop = True
    if save_crop:
        index = 0
        base = "collected_images/"
        while True:
            if not os.path.exists(os.path.join(base, f"run_{index}")):
                os.mkdir(os.path.join(base, f"run_{index}"))
                os.mkdir(os.path.join(base, f"run_{index}", "labels"))
                os.mkdir(os.path.join(base, f"run_{index}", "images"))
                path_to_save = os.path.join(base, f"run_{index}")
                break
            index += 1

    image_detector = YoloDetector(model_ckpt_path, 1920)
    process_image = ProcessImage(class_names, whitelisted_classes, save_crops=save_crop,
                                 save_location=path_to_save)

    if user_args.method == "screen":
        test_on_video(process_image, image_detector,
                      "rendered.mp4")
    # else:
    #     test_on_video(model, data["names"], user_args.video)
