"""
Collect data from a video using the model and the image cropper.
"""
import os.path

from yolov8.dataset_handling.dataset_stats import overall_healthcheck
from yolov8.DataCollection.random_image_cropper import ImageCropper
from yolov8.detectors import YoloDetector
from yolov8.DataCollection.save_crops import save_one
from yolov8.random_utils.draw_on_image import *
from yolov8.random_utils.helpful_functions import read_class_names_from_yaml


def process_video(
    path_to_video: str,
    save_location: str,
    frame_skip_factor: int = 3,
    show_output: bool = True,
    base_file_name="auto_collected",
) -> None:
    cap = cv2.VideoCapture(path_to_video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_detector = YoloDetector(model_ckpt_path, width)
    image_cropper = ImageCropper(whitelisted_classes)

    if show_output:
        class_names = image_detector.get_class_names()
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            boxes = image_detector.get_results_as_list(frame)
            for crop in image_cropper.get_crops(frame, boxes):
                save_one(crop[0], crop[1], save_location, base_file_name)
            if show_output:
                for box in boxes:
                    if box["conf"] > 0.5:
                        draw_custom_rectangle(frame, box["coords"])
                        draw_rectangle_name(
                            frame,
                            box["coords"],
                            class_names[box["class_index"]],
                            box["conf"],
                        )
                cv2.imshow("Processed Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        for _ in range(frame_skip_factor - 1):
            cap.read()
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model_ckpt_path = "../runs/detect/train13/weights/best.pt"

    number_of_occurrences = overall_healthcheck("../datasets/original_dataset")
    class_names = read_class_names_from_yaml("../datasets/original_dataset/data.yaml")
    whitelisted_classes = [
        class_names.index(key)
        for key, item in number_of_occurrences.items()
        if item < 100
    ]

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

    process_video(
        "../auxiliary_files/rendered.mp4",
        frame_skip_factor=3,
        save_location=path_to_save,
    )
