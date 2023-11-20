import os
import random
import time

import cv2
from numpy import ndarray
from yolov8.random_utils.label_transposing import *
from random_utils.draw_on_image import *


class ProcessImage:
    def __init__(self, classes_names, whitelisted_classes, save_location="collected_images",
                 crop_size=640, display_boxes=True, save_crops=False):
        self.crop_size = crop_size
        self.classes_names = classes_names
        self.original_frame = None
        self.frame_to_draw_on = None
        self.whitelisted_classes = whitelisted_classes
        self.last_saved_picture_time = time.time()
        self.save_location = save_location
        self.save_prefix = "AUTO_COLLECTED_PICTURE_"
        self.save_extension = ".png"
        self.display_boxes = display_boxes

        self.last_saved_picture_index = self.__get_last_picture_index()

    def __set_current_frames(self, frame):
        self.original_frame = frame.copy()
        self.frame_to_draw_on = frame.copy()

    def __update_last_saved_picture(self):
        self.last_saved_picture_time = time.time()

    def __increase_saved_picture_index(self):
        self.last_saved_picture_index += 1

    def __get_last_picture_index(self):
        index = 0
        while True:
            if os.path.exists(os.path.join(self.save_location, self.save_prefix + str(index) +
                                                               self.save_extension)):
                index += 1
            else:
                break
        return index

    def __split_whitelisted_and_not_whitelisted(self, boxes):
        whitelisted = []
        not_whitelisted = []
        index = 0
        for detection in boxes:
            conf = detection.conf.item()
            if conf < 0.5:
                continue
            class_index = int(detection.cls.item())
            x1, y1, x2, y2 = map(int, detection.xyxy.tolist()[0])
            box_data = {
                "coords": [x1, y1, x2, y2],
                "confidence": conf,
                "class_index": class_index,
                "id": index
            }
            if class_index not in self.whitelisted_classes:
                not_whitelisted.append(box_data)
            else:
                whitelisted.append(box_data)
            if self.display_boxes:
                self.frame_to_draw_on = draw_custom_rectangle(self.frame_to_draw_on,
                                                              coords=[x1, y1, x2, y2])
                self.frame_to_draw_on = draw_rectangle_size(self.frame_to_draw_on,
                                                            coords=[x1, y1, x2, y2])
                self.frame_to_draw_on = (
                    draw_rectangle_name(self.frame_to_draw_on, coords=[x1, y1, x2, y2],
                                        class_name=self.classes_names[class_index], conf=conf))
            index += 1
        return whitelisted, not_whitelisted

    def process_frame(self, frame: ndarray, results) -> ndarray:
        self.__set_current_frames(frame)
        # boxes = boxes[0].boxes.xyxy.tolist()
        if not len(results) > 0:
            return frame
        whitelisted, not_whitelisted = self.__split_whitelisted_and_not_whitelisted(
            results[0].boxes)
        if len(whitelisted) > 0:
            groups_of_boxes = self.__group_detections_for_cropping(whitelisted, not_whitelisted)
            for group in groups_of_boxes:
                off_x1, off_y1, off_x2, off_y2 = self.__get_crop_for_box(group[
                                                                             "combined_coords"],
                                                                         offset=True)

                cropped_image = self.original_frame[off_y1:off_y2, off_x1:off_x2]
                translated_detections = self.__translate_detection_coordinates(group[
                                                                                   "used_boxes"],
                                                                               [off_x1, off_y1,
                                                                                off_x2,
                                                                                off_y2])
                if time.time() - self.last_saved_picture_time > 1:
                    self.save_cropped_image_and_label(cropped_image, translated_detections)
                    if self.display_boxes:
                        self.frame_to_draw_on = draw_custom_rectangle(self.frame_to_draw_on,
                                                                      group["combined_coords"])
                        self.frame_to_draw_on = draw_custom_rectangle(self.frame_to_draw_on,
                                                                      [off_x1, off_y1, off_x2,
                                                                       off_y2], color=(0, 0, 255))
            self.__update_last_saved_picture()
        return self.frame_to_draw_on

    def __get_crop_for_box(self, box_coordinates, offset=False):
        width = self.original_frame.shape[1]
        height = self.original_frame.shape[0]
        x1, y1, x2, y2 = map(int, box_coordinates)
        max_x1_offset_point = x2 - self.crop_size if x2 - self.crop_size >= 0 else 0
        max_y1_offset_point = y2 - self.crop_size if y2 - self.crop_size >= 0 else 0
        max_x2_offset_point = x1 + self.crop_size if x1 + self.crop_size <= width else width
        max_y2_offset_point = y1 + self.crop_size if y1 + self.crop_size <= height else height

        if offset:
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

        # TODO Implement cropping without offset
        # if not offset:
        #     center_x=int((x2-x1)//2)
        #     center_y = int((y2 - y1) // 2)

        return [x1_new, y1_new, x2_new, y2_new]

    def __translate_detection_coordinates(self, detection_boxes_coordinates, crop_coordinates):
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

    def save_cropped_image_and_label(self, cropped_image, detections_coordinates):
        img_path = os.path.join(self.save_location, "images", self.save_prefix + str(
            self.last_saved_picture_index) + self.save_extension)
        txt_path = os.path.join(self.save_location, "labels", self.save_prefix + str(
            self.last_saved_picture_index) + ".txt")
        cv2.imwrite(img_path, cropped_image)

        with open(txt_path, 'a') as file:
            for detection in detections_coordinates:
                x_center, y_center, width, height = corner_values_to_yolo(detection["coords"])
                file.write(f"{int(detection['class_index'])} {x_center} {y_center} {width}"
                           f" {height}\n")
        self.__increase_saved_picture_index()

    # TODO this needs a better implementation
    def __group_detections_for_cropping(self, whitelisted, not_whitelisted):
        used = []
        groups = []

        def combine_two_squares(first_box, second_box):
            f_x1, f_y1, f_x2, f_y2 = first_box
            s_x1, s_y1, s_x2, s_y2 = second_box
            x1_max, y1_max, x2_max, y2_max = (int(min(f_x1, s_x1)), int(min(f_y1, s_y1)),
                                              int(max(f_x2, s_x2)), int(max(f_y2, s_y2)))
            return x1_max, y1_max, x2_max, y2_max

        # Step 1: Group whitelisted signs into the biggest possible groups, eve if some are
        # duplicate
        used_whitelisted = []
        for first_box in whitelisted:
            if first_box["id"] in used_whitelisted:
                continue
            f_x1, f_y1, f_x2, f_y2 = first_box["coords"]
            used_whitelisted.append(first_box["id"])
            current_group = {
                "used_boxes": [first_box],
                "combined_coords": [f_x1, f_y1, f_x2, f_y2]
            }
            for second_box in whitelisted:
                if second_box["id"] == first_box["id"]:
                    continue
                x1_max, y1_max, x2_max, y2_max = combine_two_squares(current_group[
                                                                         "combined_coords"],
                                                                     second_box[
                                                                         "coords"])
                if x2_max - x1_max <= 640 and y2_max - y1_max <= 640:
                    current_group["used_boxes"].append(second_box)
                    current_group["combined_coords"] = [x1_max, y1_max, x2_max, y2_max]
                    used_whitelisted.append(second_box["id"])
            groups.append(current_group)
            if len(used_whitelisted) == len(whitelisted):
                break

        # Step 2: Add any not whitelisted signs to the created groups if possible.
        for box in not_whitelisted:
            for group in groups:
                x1_max, y1_max, x2_max, y2_max = combine_two_squares(box["coords"],
                                                                     group["combined_coords"])
                if x2_max - x1_max <= 640 and y2_max - y1_max <= 640:
                    group["combined_coords"] = [x1_max, y1_max, x2_max, y2_max]
                    group["used_boxes"].append(box)

        return groups
