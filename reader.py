import cv2
import numpy as np
import os
import random


class Dataset:
    def __init__(self, images, labels):
        self._images = np.array(images)
        self._labels = np.array(labels)

        # assert the number of image and labels are same
        assert len(images) == len(labels)
        self._num_examples = len(images)
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_expamles(self):
        return self._num_examples

    def _shuffle_data(self):
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]

    def next_batch(self, batch_size, shuffle=True):
        # shuffle the images
        if shuffle and self._index_in_epoch == 0:
            self._shuffle_data()

        if batch_size >= self._num_examples:
            return self._images, self._labels

        start = self._index_in_epoch

        if start + batch_size > self._num_examples:
            rest_part_images = self._images[start:]
            rest_part_labels = self._labels[start:]

            # only shuffle will concatenate the new part data
            # the last batch may be smaller than batch size when shuffle = False
            # this benifits for the batch evaluation
            if shuffle:
                end = start + batch_size - self._num_examples
                self._shuffle_data()
                new_part_images = self._images[0:end]
                new_part_labels = self._labels[0:end]
                batch_images = np.concatenate(
                    (rest_part_images, new_part_images), axis=0)
                batch_labels = np.concatenate(
                    (rest_part_labels, new_part_labels), axis=0)
                self._index_in_epoch = end
            else:
                self._index_in_epoch = 0
                batch_images = rest_part_images
                batch_labels = rest_part_labels
        else:
            batch_images = self._images[start:start + batch_size]
            batch_labels = self._labels[start:start + batch_size]
            self._index_in_epoch = (start + batch_size) % self._num_examples

        return batch_images, batch_labels


class Reader:
    def __init__(self, data_dir, config):
        def _read_an_image(filename):
            img = cv2.imread(filename)
            img = cv2.resize(img[-150:], (200, 66))
            # BGR space to YUV space
            img = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
            return img

        def _read_all_data(data_dir):
            images_filename = os.path.join(data_dir, "images.npy")
            labels_filename = os.path.join(data_dir, "labels.npy")
            if os.path.exists(images_filename) and os.path.exists(
                    labels_filename):
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
                images = list(map(_read_an_image, filenames))
                labels = np.array(labels)
                # save the data
                np.save(images_filename, images)
                np.save(labels_filename, labels)

            return images, labels

        def _split_dataset(images, labels, train_prop, validation_prop, seed):
            data = list(zip(images, labels))

            # Random shuffle the data with specific seed for split the dataset
            # It makes sure that each time the split is same
            # You can pass and different seed for a different split
            random.seed(config.seed)
            random.shuffle(data)

            num_train = int(len(data) * config.train_prop)
            num_validation = int(len(data) * config.validation_prop)
            num_test = len(data) - num_train - num_validation

            # split the data
            train_data = data[:num_train]
            validation_data = data[num_train:num_train + num_validation]
            test_data = data[-num_test:]

            return train_data, validation_data, test_data

        images, labels = _read_all_data(data_dir)

        if config.train_prop + config.validation_prop >= 1:
            print(
                'Error params: the sum of train and validation proportion can NOT larger than 1.0'
            )
            exit(1)

        train_data, validation_data, test_data = _split_dataset(
            images, labels, config.train_prop, config.validation_prop,
            config.seed)

        train_images, train_labels = zip(*train_data)
        self._train = Dataset(train_images, train_labels)

        validation_images, validation_labels = zip(*validation_data)
        self._validation = Dataset(validation_images, validation_labels)

        test_images, test_labels = zip(*test_data)
        self._test = Dataset(test_images, test_labels)

    @property
    def train(self):
        return self._train

    @property
    def validation(self):
        return self._validation
    
    @property
    def test(self):
        return self._test
