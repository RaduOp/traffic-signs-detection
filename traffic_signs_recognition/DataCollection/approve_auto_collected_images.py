"""
This file contains a simple script that cycles through images and labels from a folder. It will
display the labels on the image and if you press "d" (you can remap this) it will delete the image.

It's great for images collected automatically by the model as they are not very accurate at the
moment.

It also helps avoid the slow process of deleting an image that's been uploaded to the dataset on
roboflow. It takes roughly 10-20 seconds to delete an image on Roboflow.
"""
import os

import cv2

from traffic_signs_recognition.random_utils.draw_on_image import draw_custom_rectangle


def check_collected_images(path_to_folder: str, delete_key: str, quit_key: str):
    for image_name in os.listdir(os.path.join(path_to_folder, "images")):
        current_picture = os.path.join(path_to_folder, "images", image_name)
        current_label = os.path.join(path_to_folder, "labels", image_name[:-4] + ".txt")

        image = cv2.imread(current_picture)

        with open(current_label, "r", encoding="utf-8") as label_file:
            for line in label_file:
                label = [float(el) for el in line.strip().split(" ")]
                image = draw_custom_rectangle(image, label[1:])

        cv2.imshow("img", image)
        key = cv2.waitKey(0)
        if key == ord(quit_key):
            break
        if key == ord(delete_key):
            if os.path.exists(current_label) and os.path.exists(current_picture):
                os.remove(current_picture)
                os.remove(current_label)
                print(
                    f"Image and label deleted:  {image_name}, {image_name[:-4]} + .txt"
                )
            else:
                print(
                    f"Delete failed, label or image is missing:{image_name} {image_name[:-4]}.txt"
                )

        else:
            print("Image approved: ", image_name)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    DELETE_KEY = "d"
    QUIT_KEY = "q"
    LOCATION = "../collected_images/run_1"
    check_collected_images(LOCATION, DELETE_KEY, QUIT_KEY)
