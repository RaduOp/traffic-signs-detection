import os

import numpy as np

from yolov8.database_health_check import read_class_names
from yolov8.random_utils.label_transposing import *
import cv2
from random_utils.label_transposing import yolo_to_corner_values


def display_multiple_pictures(img_list, image_names, class_name):
    # Calculate the dimensions of the grid
    cols = len(img_list)
    max_height = max(img.shape[0] for img in img_list)
    max_width = sum(img.shape[1] for img in img_list)

    # Create an empty canvas with a white background
    canvas = np.ones((max_height, max_width, 3), dtype=np.uint8) * 255

    # Paste each image onto the canvas
    current_width = 0
    for img in img_list:
        h, w = img.shape[:2]
        canvas[:h, current_width:current_width + w] = img
        current_width += w

    print("NEW SET_____________________________: ", class_name)
    for name in image_names:
        print(name)

    cv2.imshow(class_name, canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def check_labels_from_dataset(path_to_dataset: str) -> None:
    path_to_pictures: str = os.path.join(path_to_dataset, "images")
    all_images: list = [picture_file for picture_file in os.listdir(path_to_pictures)]

    path_to_labels: str = os.path.join(path_to_dataset, "labels")
    all_labels: list = [txt_file for txt_file in os.listdir(path_to_labels)]
    class_names = read_class_names("datasets/data.yaml")
    for index in range(72):

        current_number_of_pictures = 0
        current_pictures = []
        image_names = []
        # index = 22
        for image_path in all_images:
            with open(os.path.join(path_to_labels, image_path[:-4] + ".txt"), 'r') as label_file:
                for line in label_file:
                    label = [el for el in line.strip().split(" ")]
                    if int(label[0]) == index:
                        x_center, y_center, width, height = float(label[1]), float(label[2]), float(
                            label[3]), float(label[4])
                        x1, y1, x2, y2 = yolo_to_corner_values([x_center, y_center, width, height])
                        image = cv2.imread(os.path.join(path_to_pictures, image_path))

                        image = image[y1:y2, x1:x2]
                        # cv2.imshow("Pic", image)
                        current_pictures.append(image)
                        image_names.append(image_path)
                        current_number_of_pictures += 1

                if len(current_pictures) >= 5:
                    display_multiple_pictures(current_pictures, image_names,
                                              class_name=class_names[index])
                    current_pictures = []
                    image_names = []

        if len(current_pictures) > 0:
            display_multiple_pictures(current_pictures, image_names, class_name=class_names[index])
        print(current_number_of_pictures)


def check_collected_images(path_to_collected_images):
    path_to_pictures: str = os.path.join(path_to_collected_images, "images")
    all_images: list = [picture_file for picture_file in os.listdir(path_to_pictures)]

    path_to_labels: str = os.path.join(path_to_collected_images, "labels")
    all_labels: list = [txt_file for txt_file in os.listdir(path_to_labels)]
    for image_name in all_images:
        image = cv2.imread(os.path.join(path_to_pictures, image_name))
        with open(os.path.join(path_to_labels, image_name[:-4] + ".txt"), 'r') as label_file:
            for line in label_file:
                label = [float(el) for el in line.strip().split(" ")]
                x1, y1, x2, y2 = map(int, yolo_to_corner_values(label[1:]))
                image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.imshow("img", image)
        key = cv2.waitKey(0)
        if key == ord('q'):
            # Continue to the next image
            continue
        elif key == ord('d'):
            # Delete image and txt file
            os.remove(os.path.join(path_to_pictures, image_name))
            os.remove(os.path.join(path_to_labels, image_name[:-4] + ".txt"))
            print(f"Deleted: {image_name} and {image_name[:-4]}.txt")
        else:
            print("Invalid key pressed. Press 'q' to continue or 'd' to delete.")


# check_labels_from_dataset("datasets/valid")
check_collected_images("collected_images/run_0")
