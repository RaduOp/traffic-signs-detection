from typing import Union

import cv2
import numpy as np
from label_transposing import yolo_to_corner_values


def check_coordinates_type(coords: list[Union[int, float]]):
    if coords[0] < 1 and coords[1] < 1 and coords[2] < 1 and coords[3] < 1:
        return yolo_to_corner_values(coords)
    return coords


def draw_custom_rectangle(frame: np.ndarray, coords: list[Union[int, float]], color: tuple[int,
int, int] = (0, 255, 0),
                          thickness=1):
    x1, y1, x2, y2 = check_coordinates_type(coords)

    cv2.rectangle(frame, (x1, y1), (x2, y2),
                  color, thickness)
    return frame


def draw_rectangle_size(frame: np.ndarray, coords: list[int], color: tuple[int] = (255, 0, 0),
                        thickness: int = 1, font_scale: float = 0.5):
    x1, y1, x2, y2 = check_coordinates_type(coords)
    frame = cv2.putText(frame, str(y2 - y1), (x2 - 50, y1 + ((y2 - y1) // 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color, 2)

    frame = cv2.putText(frame, str(x2 - x1), (x1 + ((x2 - x1) // 2), y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        color, thickness)

    return frame


def draw_rectangle_name(frame: np.ndarray, coords: list[int], class_name: str, conf: float,
                        color: tuple[int] = (255, 0, 0), thickness: int = 1,
                        font_scale: float = 0.5):
    x1, y1, x2, y2 = check_coordinates_type(coords)
    cv2.putText(frame, f"{class_name}, {round(conf, 2)}", (x1, y1 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale=font_scale,
                color=color,
                thickness=thickness)

    return frame
