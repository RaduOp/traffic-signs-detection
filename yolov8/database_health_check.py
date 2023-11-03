import os
import yaml
import time
import matplotlib.pyplot as plt


def map_pictures_to_labels(path_to_dataset: str) -> dict:
    path_to_pictures: str = os.path.join(path_to_dataset, "images")
    path_to_labels: str = os.path.join(path_to_dataset, "labels")
    mapped_pictures = {

    }

    return mapped_pictures


def read_all_dataset_files(path_to_dataset: str) -> (list[str], list[str], str, str, list[str]):
    path_to_pictures: str = os.path.join(path_to_dataset, "train", "images")
    all_pictures: list = [picture_file[:-4] for picture_file in os.listdir(path_to_pictures)]

    path_to_labels: str = os.path.join(path_to_dataset, "train", "labels")
    all_labels: list = [txt_file[:-4] for txt_file in os.listdir(path_to_labels)]

    with open(os.path.join(path_to_dataset, "data.yaml"), "r") as f:
        class_names: list = yaml.safe_load(f)['names']
    return all_pictures, all_labels, path_to_pictures, path_to_labels, class_names


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
    all_pictures, all_labels, _, path_to_labels, class_names = read_all_dataset_files(
        path_to_dataset)

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


def split_dataset(path_to_dataset: str, path_to_save_split: str = "",
                  valid_split=30):
    all_pictures, all_labels, path_to_pictures, path_to_labels, \
        class_names = read_all_dataset_files(path_to_dataset)
    number_of_occurrences = dataset_health_check(path_to_dataset)
    files_to_move = []
    ideal_split_values = [0] * len(class_names)
    for index, item in enumerate(number_of_occurrences):
        ideal_split_values[index] = int(item * valid_split / 100)


def run_health_check():
    dataset_health_check('datasets')


# start = time.time()
# run_health_check()
# print(time.time() - start)

split_dataset(('datasets'))
