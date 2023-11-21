"""
This file contains a collection of methods meant for displaying rectangles or text on an image.
It uses the default cv2 methods (rectangle and putText) but adds a little bit of customization.
They are created for visualization purposes and are great for debugging.
"""

from typing import Union

import cv2
import numpy as np
from label_transposing import check_coordinates_type


def draw_custom_rectangle(frame: np.ndarray, coords: list[Union[int, float]],
                          color: tuple[int, int, int] = (0, 255, 0), thickness=1) -> np.ndarray:
    """
    Draw a rectangle on an image at given coordinates. It uses the cv2.rectangle method but does
    checks for coordinates and is customized beforehand. Might add additional steps in the future.

    :param frame: the image you want to draw on
    :param coords: coordinates in yolo or corner format
    :param color: in case you want to change the preset color
    :param thickness: border thickness
    :return: image with drawn rectangle
    """
    x1, y1, x2, y2 = check_coordinates_type(coords)

    cv2.rectangle(frame, (x1, y1), (x2, y2),
                  color, thickness)
    return frame


def draw_rectangle_size(frame: np.ndarray, coords: list[int], color: tuple[int] = (255, 0, 0),
                        thickness: int = 1, font_scale: float = 0.5) -> np.ndarray:
    """
    Draw the sizes of the rectangle (in pixels). It uses the cv2.putText method to draw line
    sizes in pixels, does checks for coordinates and is customized (color, font, font_scale,
    thickness).

    :param frame: image to draw on
    :param coords: coordinates in yolo or corner format
    :param color: in case you want to change the preset color
    :param thickness: text thickness
    :param font_scale: font size
    :return: image with drawn rectangle sizes (you might want to draw the rectangle first)
    """

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
                        font_scale: float = 0.5) -> np.ndarray:
    """
    Write a name next to a rectangle. Very useful for class names, does checks for coordinates,
    and is customized (color, font size, and thickness). Used mostly for detection boxes names.
    :param frame: image to draw on
    :param coords: coordinates in yolo or corner format
    :param class_name: the name you want to write
    :param conf: confidence for detection
    :param color: in case you want to change the preset color
    :param thickness: text thickness
    :param font_scale: text size
    :return: image with name drawn for the given rectangle (you might want to draw the rectangle
    first)
    """
    x1, y1, x2, y2 = check_coordinates_type(coords)
    cv2.putText(frame, f"{class_name}, {round(conf, 2)}", (x1, y1 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale=font_scale,
                color=color,
                thickness=thickness)

    return frame
