"""
This file is meant for live testing of the trained model.
"""
import random
import time

import cv2
import mss
import yaml
import numpy as np
from ultralytics import YOLO
import argparse
from numpy import ndarray
from database_health_check import dataset_health_check
from use_model import ImageDetector


def test_on_video(model: YOLO, names: list, path_to_video: str) -> None:
    cap = cv2.VideoCapture(path_to_video)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = process_frame(frame, model, names)
            cv2.imshow('Processed Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def test_live_on_screen_with_multiple_monitors(image_detector: ImageDetector) -> None:
    with mss.mss() as sct:
        # Get information of monitor 2
        monitor_number = 1
        mon = sct.monitors[monitor_number]

        # The screen part to capture
        monitor = {
            "top": mon["top"],
            "left": mon["left"],
            "width": mon["width"],
            "height": mon["height"],
            "mon": monitor_number,
        }

        # Grab the data
        while True:
            start = time.time()
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            frame = image_detector.process_frame(frame)
            cv2.imshow('Processed Frame', frame)
            print("Time per frame: ", time.time() - start)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    # output = "sct-mon{mon}_{top}x{left}_{width}x{height}.png".format(**monitor)
    # Save to the picture file
    # mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)


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
    model_ckpt_path = 'runs/detect/train18/weights/best.pt'

    with open(dataset_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    model = YOLO(model_ckpt_path)
    number_of_occurrences = dataset_health_check("datasets/train")
    print(number_of_occurrences)
    whitelisted_classes = [index for index, item in enumerate(number_of_occurrences) if
                           item < 60]
    image_detector = ImageDetector(classes_names=data["names"], model=model,
                                   whitelisted_classes=whitelisted_classes)
    for name, number in zip(data["names"], number_of_occurrences):
        print(name, ": ", number)
    if user_args.method == "screen":
        test_live_on_screen_with_multiple_monitors(image_detector)
    else:
        test_on_video(model, data["names"], user_args.video)
