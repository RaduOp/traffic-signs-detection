import os
from collections import Counter, OrderedDict

from matplotlib import pyplot as plt

from yolov8.random_utils.helpful_functions import read_class_names_from_yaml
import cv2
import pandas as pd


def plot_on_barchart(occurrences_dict: dict, title: str, save_path: str = None) -> None:
    sorted_data = dict(sorted(occurrences_dict.items(), key=lambda item: item[1]))
    sorted_class_names, sorted_number_of_occurrences = zip(*sorted_data.items())

    plt.figure(figsize=(19.20, 10.80), dpi=100)
    bars = plt.barh(sorted_class_names, sorted_number_of_occurrences, color="#00cda6")
    for bar, count in zip(bars, sorted_number_of_occurrences):
        plt.annotate(
            str(count),
            xy=(count, bar.get_y() + bar.get_height() / 2),
            va="center",
            fontsize=8,
        )

    plt.xlabel("Number of Occurrences")
    plt.ylabel("Class Names")
    plt.title(title)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def overall_healthcheck(path_to_dataset: str, split: str = "all", plot: bool = False):
    try:
        class_names = read_class_names_from_yaml(
            os.path.join(path_to_dataset, "data.yaml")
        )
    except FileNotFoundError:
        raise FileNotFoundError("Yaml file not found")
    try:
        df = pd.read_csv(os.path.join(path_to_dataset, "dataset.csv"))
    except FileNotFoundError:
        raise FileNotFoundError("CSV file not found.")

    if split == "all":
        current_split = df["label"].tolist()
    else:
        current_split = df[df["split"] == split]["label"].tolist()

    occurrences_counter = Counter(current_split)

    occurrences_dict = {
        label: occurrences_counter[index] if index in occurrences_counter else 0
        for index, label in enumerate(class_names)
    }
    if plot:
        plot_on_barchart(
            occurrences_dict,
            f"Healthcheck: {split}",
            save_path=f"../datasets/original_dataset/{split}.jpg",
        )

    return occurrences_dict


if __name__ == "__main__":
    print(overall_healthcheck("../datasets/original_dataset"))
