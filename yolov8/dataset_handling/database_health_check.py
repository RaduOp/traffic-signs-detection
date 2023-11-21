import os
import yaml
import time
import matplotlib.pyplot as plt


def read_dataset_as_dict(path_to_dataset: str) -> dict:
    dataset_dict = {}
    path_to_pictures: str = os.path.join(path_to_dataset, "images")
    all_images: list = [picture_file for picture_file in os.listdir(path_to_pictures)]

    path_to_labels: str = os.path.join(path_to_dataset, "labels")
    all_labels: list = [txt_file for txt_file in os.listdir(path_to_labels)]

    for image in all_images:
        dataset_dict[image[:-4]] = []
        if image[:-4] + ".txt" not in all_labels:
            print("Image has not label: ", image)
            continue
        with open(os.path.join(path_to_labels, image[:-4] + ".txt"), 'r') as label_file:
            for line in label_file:
                label = [el for el in line.strip().split(" ")]
                dataset_dict[image[:-4]].append(label)

    return dataset_dict


def read_class_names(path_to_data_file):
    with open(path_to_data_file, "r") as f:
        class_names: list = yaml.safe_load(f)['names']
    return class_names


def read_all_dataset_files(path_to_dataset: str) -> (list[str], list[str], str, str, list[str]):
    path_to_pictures: str = os.path.join(path_to_dataset, "images")
    all_pictures: list = [picture_file[:-4] for picture_file in os.listdir(path_to_pictures)]

    path_to_labels: str = os.path.join(path_to_dataset, "labels")
    all_labels: list = [txt_file[:-4] for txt_file in os.listdir(path_to_labels)]

    return all_pictures, all_labels, path_to_pictures, path_to_labels


def plot_on_barchart(class_names: list[str], number_of_occurrences: list[int], title: str) -> None:
    sorted_data = sorted(zip(class_names, number_of_occurrences), key=lambda x: x[1])
    sorted_class_names, sorted_number_of_occurrences = zip(*sorted_data)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(sorted_class_names, sorted_number_of_occurrences, color='lightgreen')
    for bar, count in zip(bars, sorted_number_of_occurrences):
        plt.annotate(str(count), xy=(count, bar.get_y() + bar.get_height() / 2), va='center',
                     fontsize=8)

    plt.xlabel('Number of Occurrences')
    plt.ylabel('Class Names')
    plt.title(title)

    plt.tight_layout()
    plt.show()


def dataset_health_check(path_to_dataset: str, show_plot=False) -> list:
    all_pictures, all_labels, _, path_to_labels = read_all_dataset_files(
        path_to_dataset)
    class_names = read_class_names("datasets/data.yaml")
    number_of_occurrences = [0] * len(class_names)

    for image_name in all_pictures:
        if image_name in all_labels:
            with open(os.path.join(path_to_labels, image_name + ".txt"), 'r') as label_file:
                for line in label_file:
                    parts = line.strip().split()
                    number_of_occurrences[int(parts[0])] += 1
        else:
            print(f'Image missing labels file: {image_name}')
    if show_plot:
        plot_on_barchart(class_names, number_of_occurrences, "Dataset healthcheck")
    return number_of_occurrences
