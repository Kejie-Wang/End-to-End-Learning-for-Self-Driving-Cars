import argparse
import os

import tensorflow as tf

from model import Nivdia_Model
import reader


def batch_eval(target, data, x_image, y, keep_prob, batch_size, sess):
    value = 0
    batch_num = (data.num_expamles + batch_size - 1) // batch_size
    for i in range(batch_num):
        batch_x, batch_y = data.next_batch(batch_size, shuffle=False)
        res = sess.run(
            target, feed_dict={
                x_image: batch_x,
                y: batch_y,
                keep_prob: 1.0
            })
        value += res * len(batch_x)

    return value / data.num_expamles


def test():
    x_image = tf.placeholder(tf.float32, [None, 66, 200, 3])
    y = tf.placeholder(tf.float32, [None, 1])
    keep_prob = tf.placeholder(tf.float32)

    model = Nivdia_Model(x_image, y, keep_prob, FLAGS, False)

    # dataset reader
    dataset = reader.Reader(FLAGS.data_dir, FLAGS)

    # model saver used to resore model from model dir
    saver = tf.train.Saver()

    with tf.Session() as sess:
        path = tf.train.latest_checkpoint(FLAGS.model_dir)
        if not (path is None):
            saver.restore(sess, path)
        else:
            print("There is not saved model in the directory of model.")
        loss = batch_eval(model.loss, dataset.test, x_image, y, keep_prob, 500,
                          sess)
        print("Loss (MSE) in test dataset:", loss)
        mae = batch_eval(model.mae, dataset.test, x_image, y, keep_prob, 500,
                         sess)
        print("MAE in test dataset: ", mae)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default=os.path.join('.', 'driving_dataset'),
        help='Directory of data')
    parser.add_argument(
        '--model_dir',
        type=str,
        default=os.path.join('.', 'saved_model'),
        help='Directory of saved model')

    FLAGS, unparsed = parser.parse_known_args()
    test()
