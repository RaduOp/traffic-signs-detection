from typing import Union


def yolo_to_corner_values(coordinates: list[Union[int, float]], image_height: int = 640,
                          image_width: int = 640):
    center_x, center_y, width, height = coordinates

    x_min = int((center_x - width / 2) * image_width)
    y_min = int((center_y - height / 2) * image_height)
    x_max = int((center_x + width / 2) * image_width)
    y_max = int((center_y + height / 2) * image_height)

    return x_min, y_min, x_max, y_max


def corner_values_to_yolo(coordinates: list[Union[int, float]], image_height: int = 640,
                          image_width: int = 640):
    x_min, y_min, x_max, y_max = coordinates
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min

    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height

    return x_center, y_center, width, height
