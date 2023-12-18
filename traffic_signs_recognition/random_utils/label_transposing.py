"""
A collection of methods that help with coordinates transposing. Sometimes you need yolo format,
and sometimes you don't, these methods make it easier.
"""

from typing import Union


def check_coordinates_type(coords: list[Union[int, float]]) -> list[int]:
    """
    This function checks the coordinates for their type (yolo/corner format) and always returns
    them in the corner format so drawing on image can be easier

    :param coords: coordinates from label file, yolo or corner format
    :return: [x1,y1,x2,y2]
    """
    if coords[0] < 1 and coords[1] < 1 and coords[2] < 1 and coords[3] < 1:
        return yolo_to_corner_values(coords)
    return coords


def yolo_to_corner_values(
    coordinates: list[Union[int, float]],
    image_height: int = 640,
    image_width: int = 640,
) -> list[int]:
    """
    A method that transforms yolo format coordinates into min corner max corner coordinates.

    :param coordinates: A list of coordinates in yolo format
    :param image_height: 640 by default but can be changed
    :param image_width: 640 by default but can be changed
    :return: coordinates in min corner max corner format (coords are cast to int for ease of use)
    """
    center_x, center_y, width, height = coordinates

    x_min = int((center_x - width / 2) * image_width)
    y_min = int((center_y - height / 2) * image_height)
    x_max = int((center_x + width / 2) * image_width)
    y_max = int((center_y + height / 2) * image_height)

    return [x_min, y_min, x_max, y_max]


def corner_values_to_yolo(
    coordinates: list[Union[int, float]],
    image_height: int = 640,
    image_width: int = 640,
) -> list[float]:
    """
    A method that transforms min corner max corner coordinates to yolo format.

    :param coordinates: a list of min corner max corner type coordinates
    :param image_height: 640 by default but can be changed
    :param image_width: 640 by default but can be changed
    :return: coordinates in yolo format according to image size
    """
    x_min, y_min, x_max, y_max = coordinates

    x_center = ((x_min + x_max) / 2.0) / image_width
    y_center = ((y_min + y_max) / 2.0) / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height

    return [x_center, y_center, width, height]
