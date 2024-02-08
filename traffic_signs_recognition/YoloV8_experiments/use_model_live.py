"""
This file is meant for live testing of the trained model.
"""
import cv2
import numpy as np
from mss import mss

from traffic_signs_recognition import YoloDetector
from traffic_signs_recognition.random_utils import (
    draw_custom_rectangle,
    draw_rectangle_name,
)


def test_on_video(
    image_detector: YoloDetector, video_path: str, output_path: str = "output.avi"
):
    """Test the model on a video. Saves the output."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(5)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    class_names = image_detector.get_class_names()
    i = 0
    s = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection on the frame
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

        # Write the frame with detections to the output video
        out.write(frame)

        i += 1
        if i % 1800 == 0:
            s += 30
            print(f"Seconds done: {s}")
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Release the video capture and writer objects
    cap.release()
    out.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()


def test_live_on_screen_with_multiple_monitors(image_detector: YoloDetector) -> None:
    """Test model live."""
    sct = mss(compression_level=1)
    monitor_number = 2
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
        if key == ord("n"):  # Press 'n' to switch to the next monitor
            monitor_number = (monitor_number + 1) % len(monitors)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    MODEL_CKPT = "../YOLOV8/runs/detect/train8/weights/best.pt"
    IMAGE_DETECTOR = YoloDetector(MODEL_CKPT, 1920)

    # test_live_on_screen_with_multiple_monitors(image_detector)
    test_live_on_screen_with_multiple_monitors(IMAGE_DETECTOR)
