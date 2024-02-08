import numpy as np
from ultralytics import YOLO


class YoloDetector:
    def __init__(self, path_to_model: str, image_size: int = None, verbose=True):
        self.image_size = image_size
        self.path_to_model = path_to_model
        self.model = self._init_model_from_path()
        self.verbose = verbose

    def change_image_size(self, new_img_size):
        self.image_size = new_img_size

    def _init_model_from_path(self):
        return YOLO(self.path_to_model)

    def _use_model_to_detect(self, frame: np.ndarray, image_size: int = None):
        if image_size is not None:
            return self.model.predict(frame, imgsz=image_size, verbose=self.verbose)
        if self.image_size is not None:
            return self.model.predict(
                frame, imgsz=self.image_size, verbose=self.verbose
            )

        return self.model.predict(frame, imgsz=frame.shape[1], verbose=self.verbose)

    def get_class_names(self):
        return self.model.names

    def get_raw_detection_results(self, frame: np.ndarray):
        return self._use_model_to_detect(frame)

    def get_results_as_list(self, frame: np.ndarray):
        results = self._use_model_to_detect(frame)
        results_list = []
        for index, box in enumerate(results[0].boxes):
            results_list.append(
                {
                    "coords": box.xyxy.tolist()[0],
                    "conf": box.conf.item(),
                    "class_index": int(box.cls.item()),
                    "id": index,
                }
            )
        return results_list
