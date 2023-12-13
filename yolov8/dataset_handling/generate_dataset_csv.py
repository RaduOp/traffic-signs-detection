import os

import pandas as pd


def check_integrity(path_to_folder: str):
    # TODO: Also check if filenames match.
    labels = os.listdir(os.path.join(path_to_folder, 'labels'))
    labels = [x[:-4] for x in labels]
    images = os.listdir(os.path.join(path_to_folder, 'images'))
    images = [x[:-4] for x in images]
    if labels == images:
        return True

    return False


def generate_dataset_csv(path_to_dataset: str, possible_folders: list[str] = None):
    if possible_folders is None:
        possible_folders = ['valid', 'test', 'train']

    all_data = []

    for folder_to_check in possible_folders:
        current_folder = os.path.join(path_to_dataset, folder_to_check)
        if not os.path.exists(current_folder):
            continue
        if not check_integrity(current_folder):
            print("One label or image is missing. CSV was not generated")
            return False, 0

        for file_name in os.listdir(os.path.join(current_folder, 'labels')):
            file_path = os.path.join(current_folder, 'labels', file_name)
            with open(file_path, 'r') as f:
                for line in f:
                    label = [el for el in line.strip().split(" ")]
                    label.append(os.path.join(folder_to_check))
                    label.append(os.path.join('labels', file_name))
                    label.append(os.path.join('images', file_name[:-4] + ".jpg"))
                    all_data.append(label)
    result_df = pd.DataFrame(all_data, columns=['label', 'x1', 'y1', 'x2', 'y2', 'split',
                                                'txt_path', 'img_path'])
    result_df.to_csv(path_to_dataset + "/dataset.csv", index=False)
    return True, all_data


if __name__ == '__main__':
    status, all_data = generate_dataset_csv('../datasets/original_dataset')

    if status:
        print(f'Created CSV: {len(all_data)} lines')
    else:
        print('CSV was not generated.')
