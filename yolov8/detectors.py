import numpy as np
from ultralytics import YOLO


class YoloDetector:

    def __init__(self, path_to_model: str, image_size: int):
        self.image_size = image_size
        self.path_to_model = path_to_model
        self.model = self.__init_model_from_path()

    def __init_model_from_path(self):
        return YOLO(self.path_to_model)

    def __use_model_to_detect(self, frame: np.ndarray):
        return self.model.predict(frame, imgsz=self.image_size)

    def __process_results(self, boxes):
        pass

    def get_raw_detection_results(self, frame: np.ndarray):
        return self.__use_model_to_detect(frame)

    def get_bounding_boxes(self):
        pass
