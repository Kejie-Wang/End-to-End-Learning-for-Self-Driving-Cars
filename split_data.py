import argparse
import random
import os

import cv2
import numpy as np

FLAGS = None

def read_an_image(filename):
    img = cv2.imread(filename)
    img = cv2.resize(img[-150:], (200, 66))
    # BGR space to YUV space
    img = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    return img


def read_all_data(data_dir):
    """Read all data from a directory of data.
    
    Args:
        data_dir: The directory of data.
    Returns:
        A tuple (images, lables):
          - images: A numpy array of images.
          - lables: A numpy array of lables.
    """
    images_filename = os.path.join(data_dir, "images.npy")
    labels_filename = os.path.join(data_dir, "labels.npy")
    if os.path.exists(images_filename) and os.path.exists(labels_filename):
        images = np.load(images_filename)
        labels = np.load(labels_filename)
    else:
        filenames = []
        labels = []

        data_filename = os.path.join(data_dir, 'data.txt')
        with open(data_filename) as fp:
            for line in fp:
                filename, label = line.split()
                filenames.append(os.path.join(data_dir, filename))
                labels.append([float(label) * np.pi / 180])
        images = list(map(read_an_image, filenames))
        labels = np.array(labels)
        # save the data
        np.save(images_filename, images)
        np.save(labels_filename, labels)

    return images, labels


def split_dataset(images, labels, train_prop, validation_prop, seed):
    """Split the dataset with train, validation and test set.
    
    Args:
        images: An numpy array of images.
        labels: An numpy array of labels.
        train_prop: The proportion of train data.
        validation_prop: The proportion of validation data.
        seed: The random seed used to random the dataset.

    Returns:
        A tuple (train_data, validation_data, test_data).
          - train_data: A numpy array of train data.
          - validation_data: A numpy array of validation data.
          - test_data: A numpy array of test data.
    """
    data = list(zip(images, labels))

    # Random shuffle the data with specific seed for split the dataset
    # It makes sure that each time the split is same
    # You can pass and different seed for a different split
    random.seed(seed)
    random.shuffle(data)

    num_train = int(len(data) * train_prop)
    num_validation = int(len(data) * validation_prop)
    num_test = len(data) - num_train - num_validation

    # split the data
    train_data = data[:num_train]
    validation_data = data[num_train:num_train + num_validation]
    test_data = data[-num_test:]

    return train_data, validation_data, test_data


def main():
    train_filename = os.path.join(FLAGS.data_dir, "train.npy")
    validation_filename = os.path.join(FLAGS.data_dir, "validation.npy")
    test_filename = os.path.join(FLAGS.data_dir, "test.npy")

    images, labels = read_all_data(FLAGS.data_dir)
    train_data, validation_data, test_data = split_dataset(
        images, labels, FLAGS.train_prop, FLAGS.validation_prop, FLAGS.seed)
    np.save(train_filename, train_data)
    np.save(validation_filename, validation_data)
    np.save(test_filename, test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default=os.path.join('.', 'driving_dataset'),
        help='Directory of data')
    parser.add_argument(
        '--seed',
        type=str,
        default=0,
        help='random seed to generate train, validation and test set')
    parser.add_argument(
        '--train_prop',
        type=float,
        default=0.8,
        help='The proportion of train set in all data')
    parser.add_argument(
        '--validation_prop',
        type=float,
        default=0.1,
        help='The proportion of validation set in all data')
    FLAGS, unparsed = parser.parse_known_args()

    main()
