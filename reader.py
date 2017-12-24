import numpy as np
import os


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
        train_filename = os.path.join(data_dir, "train.npy")
        validation_filename = os.path.join(data_dir, "validation.npy")
        test_filename = os.path.join(data_dir, "test.npy")

        if os.path.exists(train_filename) and os.path.exists(
                validation_filename) and os.path.exists(test_filename):
            train_data = np.load(train_filename)
            validation_data = np.load(validation_filename)
            test_data = np.load(test_filename)
        else:
            print(
                "Data does NOT exist, please check directory if exists and run split_dataset.py before train."
            )
            exit(0)

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
