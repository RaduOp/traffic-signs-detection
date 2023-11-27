"""

"""
import os
import random

import cv2
from numpy import ndarray
from yolov8.random_utils.label_transposing import corner_values_to_yolo


class ImageCropper:
    """A class that makes crops of custom sizes from an image with detections."""

    def __init__(self, whitelisted_classes, min_confidence=0.6, crop_size=640):
        """
        Initialize the ProcessImage object.

        :param whitelisted_classes: List of whitelisted class indices.
        :param min_confidence: Minimum confidence threshold for detection.
        :param crop_size: Size of the cropped images.
        """
        self.crop_size = crop_size
        self.whitelisted_classes = whitelisted_classes
        self.min_confidence = min_confidence

        # Parameters related to file saving
        self.original_frame = None

    def _set_original_frame(self, frame):
        """Save the original frame in a variable for ease of access."""
        self.original_frame = frame.copy()

    def _split_whitelisted_and_not_whitelisted(self, boxes):
        """
        Receives  list of detection boxes and splits them into whitelisted and not whitelisted
        boxes for further grouping.
        """
        whitelisted = []
        not_whitelisted = []
        for index, detection in enumerate(boxes):
            if detection["conf"] < self.min_confidence:
                continue

            if detection["class_index"] not in self.whitelisted_classes:
                not_whitelisted.append(detection)
            else:
                whitelisted.append(detection)

        return whitelisted, not_whitelisted

    def get_crops(self, frame: ndarray, boxes: list[dict]) -> list[tuple[ndarray, dict]] or list:
        """
        This method accepts an image and a list of boxes and crops one or more images based on
        the detections.

        :param frame: The original frame with no drawings on it.
        :param boxes: Detections from the image. Requires a list of dicts with these keys:
                        -coords: list of box coordinates, x1,y1,x2,y2
                        -conf: detection confidence
                        -class_index: the index of the detected class
                        -id: a unique id for the detection
        :return:
        """
        final_crops = []
        if len(boxes) < 0:
            return final_crops
        self._set_original_frame(frame)

        # Whitelisting
        whitelisted, not_whitelisted = self._split_whitelisted_and_not_whitelisted(boxes)
        if not len(whitelisted) > 0:
            return final_crops

        # Cropping
        groups_of_boxes = self._group_detections_for_cropping(whitelisted, not_whitelisted)
        for group in groups_of_boxes:
            off_x1, off_y1, off_x2, off_y2 = self._get_crop_for_box(group["combined_coords"])
            cropped_image = self.original_frame[off_y1:off_y2, off_x1:off_x2]

            translated_detections = self._detection_coords_to_crop_coords(
                group["used_boxes"], [off_x1, off_y1, off_x2, off_y2]
            )
            final_crops.append((cropped_image, translated_detections))
        return final_crops

    # TODO: maybe implement cropping without offset?
    #  Center the sign check if edges are in frame and crop?
    def _get_crop_for_box(self, box_coordinates):
        """
        Draw a box of rop_size x crop_size around the previously created group of detections and
        randomly offset the position of the box.
        """
        width = self.original_frame.shape[1]
        height = self.original_frame.shape[0]
        x1, y1, x2, y2 = map(int, box_coordinates)
        max_x1_offset_point = x2 - self.crop_size if x2 - self.crop_size >= 0 else 0
        max_y1_offset_point = y2 - self.crop_size if y2 - self.crop_size >= 0 else 0
        max_x2_offset_point = x1 + self.crop_size if x1 + self.crop_size <= width else width
        max_y2_offset_point = y1 + self.crop_size if y1 + self.crop_size <= height else height

        # TODO: make sure the crop doesn't exceed image edges in a better way
        while True:
            if random.randint(0, 1) == 0:
                x1_new = random.randint(max_x1_offset_point, x1)
                x2_new = x1_new + self.crop_size
            else:
                x2_new = random.randint(x2, max_x2_offset_point)
                x1_new = x2_new - self.crop_size

            if random.randint(0, 1) == 0:
                y1_new = random.randint(max_y1_offset_point, y1)
                y2_new = y1_new + self.crop_size
            else:
                y2_new = random.randint(y2, max_y2_offset_point)
                y1_new = y2_new - self.crop_size
            if not (x1_new < 0 or y1_new < 0 or x2_new > width or y2_new > height):
                break

        return [x1_new, y1_new, x2_new, y2_new]

    @staticmethod
    def _detection_coords_to_crop_coords(detection_boxes_coordinates, crop_coordinates):
        """Adjust the original coordinates from the bigger image to the cropped image. """
        new_boxes = []
        box_x1, box_y1, box_x2, box_y2 = map(int, crop_coordinates)
        for box in detection_boxes_coordinates:
            x1, y1, x2, y2 = map(int, box["coords"])

            new_x1 = x1 - box_x1
            new_y1 = y1 - box_y1
            new_x2 = new_x1 + (x2 - x1)
            new_y2 = new_y1 + (y2 - y1)
            box["coords"] = [new_x1, new_y1, new_x2, new_y2]
            new_boxes.append(box)
        return new_boxes

    # TODO: this needs a better implementation
    def _group_detections_for_cropping(self, whitelisted, not_whitelisted):
        """Group as many signs as possible for further cropping"""

        # Step 1: Group wl signs into the biggest possible groups, even if some are duplicate
        # If two signs are more than 640 pixels apart then they can't be grouped together and if
        # there are other signs in between there is a high chance that when cropping, those signs
        # will be in the cropped pictures. If we group them from the start we can also save their
        # labels thus avoid further work.
        used_whitelisted = []
        groups = []
        for first_box in whitelisted:
            if first_box["id"] in used_whitelisted:
                continue
            used_whitelisted.append(first_box["id"])
            current_group = {
                "used_boxes": [first_box],
                "combined_coords": first_box["coords"]
            }
            for second_box in whitelisted:
                if second_box["id"] == first_box["id"]:
                    continue
                max_x1, max_y1, max_x2, max_y2 = self._combine_two_boxes(
                    current_group["combined_coords"], second_box["coords"]
                )

                if max_x2 - max_x1 <= self.crop_size and max_y2 - max_y1 <= self.crop_size:
                    current_group["used_boxes"].append(second_box)
                    current_group["combined_coords"] = [max_x1, max_y1, max_x2, max_y2]
                    used_whitelisted.append(second_box["id"])
            groups.append(current_group)
            if len(used_whitelisted) == len(whitelisted):
                break

        # Step 2: Add any not whitelisted signs to the created groups if possible.
        for box in not_whitelisted:
            for group in groups:
                max_x1, max_y1, max_x2, max_y2 = self._combine_two_boxes(box["coords"],
                                                                         group[
                                                                             "combined_coords"])
                if max_x2 - max_x1 <= self.crop_size and max_y2 - max_y1 <= self.crop_size:
                    group["combined_coords"] = [max_x1, max_y1, max_x2, max_y2]
                    group["used_boxes"].append(box)

        return groups

    @staticmethod
    def _combine_two_boxes(first_box_coords, second_box_coords):
        """Combine the coordinates of two boxes."""
        firs_x1, first_y1, first_x2, first_y2 = first_box_coords
        second_x1, second_y1, second_x2, second_y2 = second_box_coords
        max_x1, max_y1, max_x2, max_y2 = (
            int(min(firs_x1, second_x1)), int(min(first_y1, second_y1)),
            int(max(first_x2, second_x2)), int(max(first_y2, second_y2))
        )
        return max_x1, max_y1, max_x2, max_y2
