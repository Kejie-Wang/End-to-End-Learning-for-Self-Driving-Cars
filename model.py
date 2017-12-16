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
    def inference(self, images):
        # summary first 10 images
        tf.summary.image('input', images, 10)

        with tf.name_scope('batch_norm'):
            mean, variance = tf.nn.moments(images, axes=[0, 1, 2])
            norm_images = tf.nn.batch_normalization(images, mean, variance, None, None, 1e-4)

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

        # second fully connected layer
        # 100 ===> 50
        with tf.name_scope('fc2') as scope:
            with tf.name_scope('weights'):
                weight = _weight_variable('weights', [100, 50])
                _variable_summaries(weight)
            with tf.name_scope('bias'):
                bias = _bias_variable('bias', [50])
                _variable_summaries(bias)
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1, weight) + bias)

        # third fully connected layer
        # 50 ===> 10
        with tf.name_scope('fc3') as scope:
            with tf.name_scope('weights'):
                weight = _weight_variable('weights', [50, 10])
                _variable_summaries(weight)
            with tf.name_scope('bias'):
                bias = _bias_variable('bias', [10])
                _variable_summaries(bias)
            h_fc3 = tf.nn.relu(tf.matmul(h_fc2, weight) + bias)

        # output layer
        # 10 ===> 1
        with tf.name_scope('output') as scope:
            with tf.name_scope('weights'):
                weight = _weight_variable('weights', [10, 1])
                _variable_summaries(weight)
            with tf.name_scope('bias'):
                bias = _bias_variable('bias', [1])
                _variable_summaries(bias)
            output = tf.matmul(h_fc3, weight) + bias

        return output

    def __init__(self, data, target, config):
        self._prediction = self.inference(data)
        with tf.name_scope('loss'):
            self._loss = tf.reduce_mean(tf.square(self._prediction - target))
        with tf.name_scope('optimization'):
            self._optimization = tf.train.AdamOptimizer(
                config.learning_rate).minimize(self._loss)
        # add loss summary
        tf.summary.scalar('loss', self._loss)
        with tf.name_scope('mape'):
            self._mape = tf.reduce_mean(
                tf.div(tf.abs(self._prediction - target), target))
        with tf.name_scope('mae'):
            self._mae = tf.reduce_mean(tf.abs(self._prediction - target))

    @property
    def prediction(self):
        return self._prediction

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
