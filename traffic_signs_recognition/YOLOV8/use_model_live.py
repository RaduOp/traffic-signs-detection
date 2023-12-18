"""
This file is meant for live testing of the trained model.
"""
import cv2
import numpy as np
from mss import mss
import argparse
from traffic_signs_recognition.YOLOV8.detectors import YoloDetector
from traffic_signs_recognition.random_utils import (
    draw_custom_rectangle,
    draw_rectangle_name,
)


def test_live_on_screen_with_multiple_monitors(image_detector: YoloDetector) -> None:
    sct = mss(compression_level=1)
    monitor_number = 1
    monitors = mss().monitors
    monitor = {
        "top": monitors[monitor_number]["top"],
        "left": monitors[monitor_number]["left"],
        "width": monitors[monitor_number]["width"],
        "height": monitors[monitor_number]["height"],
        "mon": monitor_number,
    }
    class_names = image_detector.get_class_names()
    while True:
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        boxes = image_detector.get_results_as_list(frame)
        if len(boxes) > 0:
            for box in boxes:
                if box["conf"] > 0.5:
                    frame = draw_custom_rectangle(frame, box["coords"])
                    frame = draw_rectangle_name(
                        frame,
                        box["coords"],
                        class_name=class_names[box["class_index"]],
                        conf=box["conf"],
                    )

        cv2.imshow("Processed Frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("n"):  # Press 'n' to switch to the next monitor
            monitor_number = (monitor_number + 1) % len(monitors)

    cv2.destroyAllWindows()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Testing the model on video or live on main "
    )

    user_args = parser.parse_args()
    return user_args


if __name__ == "__main__":
    user_args = parse_arguments()
    dataset_yaml_path = "datasets/data.yaml"
    model_ckpt_path = "../runs/detect/train17/weights/last.pt"

    image_detector = YoloDetector(model_ckpt_path, 1920)

    test_live_on_screen_with_multiple_monitors(image_detector)
