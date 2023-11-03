"""
This file is meant for live testing of the trained model.
"""

import cv2
import pyautogui
import yaml
import numpy as np
from ultralytics import YOLO
import argparse
from numpy import ndarray


def process_frame(frame: ndarray, model: YOLO, names: list) -> ndarray:
    results = model.predict(frame)

    for result in results:
        boxes = result.boxes
        for box, conf, class_name in zip(boxes.xyxy.tolist(), boxes.conf.tolist(),
                                         boxes.cls.tolist()):
            x, y, w, h = box
            x, y, w, h = int(x), int(y), int(w), int(h)
            label = f"{names[int(class_name)]}, {conf:.2f}"
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
    return frame


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


def test_live_on_screen(model: YOLO, names: list) -> None:
    screen_width, screen_height = pyautogui.size()
    screen_resolution = (screen_width, screen_height)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter("screen_recording.avi", fourcc, 20.0, screen_resolution)
    while True:
        screenshot = pyautogui.screenshot()
        frame = np.array(screenshot)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = process_frame(frame, model, names)
        cv2.imshow('Processed Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()


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
    model_ckpt_path = 'runs/detect/train17/weights/best.pt'

    with open(dataset_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    model = YOLO(model_ckpt_path)

    if user_args.method == "screen":
        test_live_on_screen(model, data["names"])
    else:
        test_on_video(model, data["names"], user_args.video)
