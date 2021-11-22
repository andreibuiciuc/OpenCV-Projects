import numpy as np
import random
import os

from PIL import Image
from sklearn.model_selection import train_test_split

generic_path = '../../model/dataset'
image_counter = 0


def rename_data():
    global generic_path, image_counter

    for i in range(1, 6):
        path_i = generic_path + '/' + str(i)
        for entry in os.scandir(path_i):
            source = path_i + '/' + entry.name
            destination = path_i + '/' + 'image_class' + str(i) + '_' + str(image_counter) + '.jpg'
            os.rename(source, destination)
            image_counter += 1


def get_image_entries(directory_path):
    entries = []
    for entry in os.scandir(directory_path):
        entries.append(entry)

    return entries


def get_image_entries_main():
    global generic_path

    entries = []
    for i in range(1, 6):
        path_i = generic_path + '/' + str(i)
        entries_i = get_image_entries(path_i)
        entries += entries_i

    return entries


def get_data(x):
    global generic_path

    X = []
    Y = []
    for entry in x:
        if '_class1_' in entry.name:
            label = 0
        elif '_class2_' in entry.name:
            label = 1
        elif '_class3_' in entry.name:
            label = 2
        elif '_class4_' in entry.name:
            label = 3
        elif '_class5_' in entry.name:
            label = 4
        image_path = generic_path + '/' + str(label+1) + '/' + entry.name
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        X.append(image_array)
        Y.append(label)

    return X, Y


def get_train_test_split():
    x_entries = get_image_entries_main()
    random.shuffle(x_entries)
    X, Y = get_data(x_entries)
    X = np.array(X)
    Y = np.array(Y)

    return train_test_split(X, Y, test_size=.33, random_state=42)


# rename_data()
