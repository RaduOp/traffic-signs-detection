"""
How scoring works.

Difficulty 1:
    - must detect signs
    - mistakes not allowed and are punished
    - strong indicator of consistency?
    - correct detection full points
    - correct class no points
Difficulty 2:
    - clear signs that are not facing the camera/are further away/are slightly obstructed
    - mistakes are allowed and not punished
    - strong indicator of good performance?
    - correct detection full points
    - correct class half points
Difficulty 3:
    - signs that are small/not clear/really skewed/obstructed
    - no detections expected here
    - if these are detected then I'm doing something good :))
    - correct detection full points
    - correct class full points
"""

import os
import cv2
import pandas as pd

from traffic_signs_recognition import YoloDetector


class BenchmarkModel:
    """
    A class used for benchmarking. Initialize this with a testing directory that con tains a
    benchmark.csv file and then run the benchmark with whatever image detector you want.
    """

    def __init__(
        self,
        test_folder,
        overlap=0.8,
        csv_name="benchmark.csv",
    ):
        self.test_folder = test_folder
        self.csv_name = csv_name
        self.overlap = overlap

        # These params are set at each run
        self.image_detector = None
        self.class_names = None
        self.score = None

    def __init_image_detector(self, image_detector):
        self.image_detector = image_detector
        self.class_names = image_detector.get_class_names()
        self.score = {"D1": 0, "D2": 0, "D3": 0, "FN": 0, "FP": 0}

    def __rectangle_overlap(self, label, detection):
        """
        Calculate the overlap percentage of two rectangles in both directions,
        label over detection and detection over label
        """
        x1_overlap = max(label[0], detection[0])
        y1_overlap = max(label[1], detection[1])
        x2_overlap = min(label[2], detection[2])
        y2_overlap = min(label[3], detection[3])

        if x1_overlap < x2_overlap and y1_overlap < y2_overlap:
            intersection_area = (x2_overlap - x1_overlap) * (y2_overlap - y1_overlap)
            rect1_area = (label[2] - label[0]) * (label[3] - label[1])
            rect2_area = (detection[2] - detection[0]) * (detection[3] - detection[1])

            overlap_percentage_1 = intersection_area / rect1_area
            overlap_percentage_2 = intersection_area / rect2_area

            if (
                overlap_percentage_1 >= self.overlap
                and overlap_percentage_2 >= self.overlap - 0.1
            ):
                # print(
                #     f"Label overlap: {overlap_percentage_1}, Detection overlap: "
                #     f"{overlap_percentage_2}"
                # )
                return True

        return False

    def __check_label_difference(self, label_index, detection_index):
        """How good was the detection based on class name:
        0: detected but completely wrong label
        1: fully correct
        2: correct category only (like 'mand')
        """
        if self.class_names[label_index] == self.class_names[detection_index]:
            return 1
        if self.class_names[label_index][:4] == self.class_names[detection_index][:4]:
            return 2

        return 0

    def __calculate_image_score(self, labels, detections):
        # Compare labels with detections and match them based on overlap.
        for _, row in labels.iterrows():
            for detection in detections:
                if self.__rectangle_overlap(
                    [row["x1"], row["y1"], row["x2"], row["y2"]], detection["coords"]
                ):
                    is_correct = self.__check_label_difference(
                        row["label"], detection["class_index"]
                    )
                    if row["difficulty"] == 1:
                        if is_correct == 1:
                            self.score["D1"] += 1
                        elif is_correct == 0:
                            self.score["D1"] += 0.33
                    elif row["difficulty"] == 2:
                        if is_correct == 1:
                            self.score["D2"] += 1
                        elif is_correct == 2:
                            self.score["D2"] += 0.5
                    elif row["difficulty"] == 3:
                        if is_correct == 1:
                            self.score["D3"] += 1
                        elif is_correct == 2:
                            self.score["D3"] += 1
                        elif is_correct == 0:
                            self.score["D3"] += 0.5

                    detections.remove(detection)
                    break
            else:
                self.score["FN"] += 1
        self.score["FP"] += len(detections)

    def display_results(self, max_score):
        print(f"Difficulty 1 score: {self.score['D1']}/{max_score[0]}")
        print(f"Difficulty 2 score: {self.score['D2']}/{max_score[1]}")
        print(f"Difficulty 3 score: {self.score['D3']}/{max_score[2]}")
        print(
            f"False Positives: {self.score['FP']}\nFalse Negatives: {self.score['FN']}"
        )

    def run_benchmark(self, image_detector):
        # Set parameters and reset the score
        self.__init_image_detector(image_detector)

        # Read the test data
        df_data = pd.read_csv(os.path.join(self.test_folder, "benchmark.csv"))
        all_images = df_data["img_name"].unique()

        # Cycle through images and use the model on each, then score the output
        for image_name in all_images:
            current_image = cv2.imread(os.path.join(self.test_folder, image_name))

            self.__calculate_image_score(
                df_data[df_data["img_name"] == image_name],
                self.image_detector.get_results_as_list(current_image),
            )

        max_score = [
            df_data[df_data["difficulty"] == 1].shape[0],
            df_data[df_data["difficulty"] == 2].shape[0],
            df_data[df_data["difficulty"] == 3].shape[0],
        ]
        self.display_results(max_score)
        return self.score, max_score


if __name__ == "__main__":
    TEST_DIR = "test_data"
    MODEL_CKPT = "../YOLOV8/runs/detect/train8/weights/best.pt"

    detector = YoloDetector(MODEL_CKPT, verbose=False)
    benchmarker = BenchmarkModel(TEST_DIR)
    benchmarker.run_benchmark(detector)
