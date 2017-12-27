import cv2
import tensorflow as tf


def _weight_variable(name, shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def _bias_variable(name, shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def _conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')


def _conv2d_transpose(x, W, stride, output_shape):
    # return tf.image.resize_images(x, [output_shape[1], output_shape[2]])
    return tf.nn.conv2d_transpose(
        x,
        W,
        output_shape=output_shape,
        strides=[1, stride, stride, 1],
        padding='VALID')


def _variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


class Nivdia_Model:
    def inference(self, images, keep_prob):
        # summary first 10 images
        tf.summary.image('input', images, 10)

        with tf.name_scope('batch_norm'):
            mean, variance = tf.nn.moments(images, axes=[0, 1, 2])
            norm_images = tf.nn.batch_normalization(images, mean, variance,
                                                    None, None, 1e-4)

        # first convolutional layer
        # 3@66x200 ===> 24@31x98
        with tf.name_scope('conv1') as scope:
            with tf.name_scope('kernel'):
                kernel = _weight_variable('weights', [5, 5, 3, 24])
                _variable_summaries(kernel)
            with tf.name_scope('bias'):
                bias = _bias_variable('bias', [24])
                _variable_summaries(bias)
            h_conv1 = tf.nn.relu(_conv2d(norm_images, kernel, 2) + bias)
        self._h_conv1 = h_conv1

        # second convolutional layer
        # 24@31x98 ===> 36@14x47
        with tf.name_scope('conv2') as scope:
            with tf.name_scope('kernel'):
                kernel = _weight_variable('weights', [5, 5, 24, 36])

                _variable_summaries(kernel)
            with tf.name_scope('bias'):
                bias = _bias_variable('bias', [36])
                _variable_summaries(bias)
            h_conv2 = tf.nn.relu(_conv2d(h_conv1, kernel, 2) + bias)
        self._h_conv2 = h_conv2

        # third convolutional layer
        # 36@14x47 ===> 48@5x22
        with tf.name_scope('conv3') as scope:
            with tf.name_scope('kernel'):
                kernel = _weight_variable('weights', [5, 5, 36, 48])
                _variable_summaries(kernel)
            with tf.name_scope('bias'):
                bias = _bias_variable('bias', [48])
                _variable_summaries(bias)
            h_conv3 = tf.nn.relu(_conv2d(h_conv2, kernel, 2) + bias)
        self._h_conv3 = h_conv3

        # fourth convolutional layer
        # 48@5x22 ===> 64@3x20
        with tf.name_scope('conv4') as scope:
            with tf.name_scope('kernel'):
                kernel = _weight_variable('weights', [3, 3, 48, 64])
                _variable_summaries(kernel)
            with tf.name_scope('bias'):
                bias = _bias_variable('bias', [64])
                _variable_summaries(bias)
            h_conv4 = tf.nn.relu(_conv2d(h_conv3, kernel, 1) + bias)
        self._h_conv4 = h_conv4

        # fifth convolutional layer
        # 64@3x20 ===> 64@1x18
        with tf.name_scope('conv5') as scope:
            with tf.name_scope('kernel'):
                kernel = _weight_variable('weights', [3, 3, 64, 64])
                _variable_summaries(kernel)
            with tf.name_scope('bias'):
                bias = _bias_variable('bias', [64])
                _variable_summaries(bias)
            h_conv5 = tf.nn.relu(_conv2d(h_conv4, kernel, 1) + bias)
        self._h_conv5 = h_conv5

        # flatted
        # 64@1x18 ===> 64x18
        h_conv5_flat = tf.reshape(h_conv5, [-1, 64 * 18])

        # first fully connected layer
        # 64x18 ===> 100
        with tf.name_scope('fc1') as scope:
            with tf.name_scope('weights'):
                weight = _weight_variable('weights', [64 * 18, 100])
                _variable_summaries(weight)
            with tf.name_scope('bias'):
                bias = _bias_variable('bias', [100])
                _variable_summaries(bias)
            h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, weight) + bias)

        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        # second fully connected layer
        # 100 ===> 50
        with tf.name_scope('fc2') as scope:
            with tf.name_scope('weights'):
                weight = _weight_variable('weights', [100, 50])
                _variable_summaries(weight)
            with tf.name_scope('bias'):
                bias = _bias_variable('bias', [50])
                _variable_summaries(bias)
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, weight) + bias)
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

        # third fully connected layer
        # 50 ===> 10
        with tf.name_scope('fc3') as scope:
            with tf.name_scope('weights'):
                weight = _weight_variable('weights', [50, 10])
                _variable_summaries(weight)
            with tf.name_scope('bias'):
                bias = _bias_variable('bias', [10])
                _variable_summaries(bias)
            h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, weight) + bias)
        h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

        # output layer
        # 10 ===> 1
        with tf.name_scope('output') as scope:
            with tf.name_scope('weights'):
                weight = _weight_variable('weights', [10, 1])
                _variable_summaries(weight)
            with tf.name_scope('bias'):
                bias = _bias_variable('bias', [1])
                _variable_summaries(bias)
            output = tf.matmul(h_fc3_drop, weight) + bias

        return output

    def inference_transpose(self):
        with tf.name_scope('conv5_transpose'):
            h_conv5_mean = tf.reduce_mean(
                self._h_conv5, axis=[3], keep_dims=True)
            h_conv5_transpose = _conv2d_transpose(
                h_conv5_mean, tf.ones([3, 3, 1, 1]), 1,
                tf.stack([self._batch_size, 3, 20, 1]))

        with tf.name_scope('conv4_transpose'):
            h_conv4_mean = tf.reduce_mean(
                self._h_conv4, axis=[3], keep_dims=True)
            h_conv4_mask = tf.multiply(h_conv4_mean, h_conv5_transpose)
            h_conv4_transpose = _conv2d_transpose(
                h_conv4_mask, tf.ones([3, 3, 1, 1]), 1,
                tf.stack([self._batch_size, 5, 22, 1]))

        with tf.name_scope('conv3_transpose'):
            h_conv3_mean = tf.reduce_mean(
                self._h_conv3, axis=[3], keep_dims=True)
            h_conv3_mask = tf.multiply(h_conv3_mean, h_conv4_transpose)
            h_conv3_transpose = _conv2d_transpose(
                h_conv3_mask, tf.ones([5, 5, 1, 1]), 2,
                tf.stack([self._batch_size, 14, 47, 1]))

        with tf.name_scope('conv2_transpose'):
            h_conv2_mean = tf.reduce_mean(
                self._h_conv2, axis=[3], keep_dims=True)
            h_conv2_mask = tf.multiply(h_conv2_mean, h_conv3_transpose)
            h_conv2_transpose = _conv2d_transpose(
                h_conv2_mask, tf.ones([5, 5, 1, 1]), 2,
                tf.stack([self._batch_size, 31, 98, 1]))

        with tf.name_scope('conv1_transpose'):
            h_conv1_mean = tf.reduce_mean(
                self._h_conv1, axis=[3], keep_dims=True)
            h_conv1_mask = tf.multiply(h_conv1_mean, h_conv2_transpose)
            h_conv1_transpose = _conv2d_transpose(
                h_conv1_mask, tf.ones([5, 5, 1, 1]), 2,
                tf.stack([self._batch_size, 66, 200, 1]))

        return h_conv1_transpose

    def __init__(self, data, target, keep_prob, config, is_train=True):
        self._batch_size = tf.shape(data)[0]
        self._prediction = self.inference(data, keep_prob)
        with tf.name_scope('loss'):
            self._loss = tf.reduce_mean(
                    tf.square(self._prediction - target))
            tf.summary.scalar('loss', self._loss)
        with tf.name_scope('mape'):
            self._mape = tf.reduce_mean(
                    tf.div(tf.abs(self._prediction - target), target))
            with tf.name_scope('mae'):
                self._mae = tf.reduce_mean(tf.abs(self._prediction - target))

        # train a model and define model optimization op
        if is_train:
            with tf.name_scope('optimization'):
                self._optimization = tf.train.AdamOptimizer(
                    config.learning_rate).minimize(self._loss)

        self._visualization_mask = self.inference_transpose()

    @property
    def prediction(self):
        return self._prediction

    @property
    def visualization_mask(self):
        return self._visualization_mask

    @property
    def loss(self):
        return self._loss

    @property
    def optimization(self):
        return self._optimization

    @property
    def mae(self):
        return self._mae

    @property
    def mape(self):
        return self._mape
