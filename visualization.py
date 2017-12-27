import argparse
import os

import cv2
import numpy as np
import tensorflow as tf

from model import Nivdia_Model
import reader

FLAGS = None


def visualize(image, mask):
    # cast image from yuv to brg.
    image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    max_val = np.max(mask)
    min_val = np.min(mask)
    mask = (mask - min_val) / (max_val - min_val)
    mask = (mask * 255.0).astype(np.uint8)
    overlay = np.copy(image) 
    overlay[:, :, 1] = cv2.add(image[:, :, 1], mask)

    return image, mask, overlay


def main():
    x_image = tf.placeholder(tf.float32, [None, 66, 200, 3])
    keep_prob = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32, [None, 1])

    model = Nivdia_Model(x_image, y, keep_prob, FLAGS, False)

    # dataset reader
    dataset = reader.Reader(FLAGS.data_dir, FLAGS)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # initialize all varibales
        sess.run(tf.global_variables_initializer())
        # restore model
        print(FLAGS.model_dir)
        path = tf.train.latest_checkpoint(FLAGS.model_dir)
        if path is None:
            print("Err: the model does NOT exist")
            exit(0)
        else:
            saver.restore(sess, path)
            print("Restore model from", path)

        batch_x, batch_y = dataset.train.next_batch(FLAGS.visualization_num,
                                                    False)
        y_pred = sess.run(
            model.prediction, feed_dict={
                x_image: batch_x,
                keep_prob: 1.0
            })
        masks = sess.run(
            model.visualization_mask,
            feed_dict={
                x_image: batch_x,
                keep_prob: 1.0
            })

    if not os.path.exists(FLAGS.result_dir):
        os.makedirs(FLAGS.result_dir)
    for i in range(FLAGS.visualization_num):
        image, mask, overlay = visualize(batch_x[i], masks[i])
        cv2.imwrite(
            os.path.join(FLAGS.result_dir, "image_" + str(i) + ".jpg"), image)
        cv2.imwrite(
            os.path.join(FLAGS.result_dir, "mask_" + str(i) + ".jpg"), mask)
        cv2.imwrite(
            os.path.join(FLAGS.result_dir, "overlay_" + str(i) + ".jpg"),
            overlay)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_dir',
        type=str,
        default=os.path.join('.', 'saved_model'),
        help='Directory of saved model')
    parser.add_argument(
        '--data_dir',
        type=str,
        default=os.path.join('.', 'driving_dataset'),
        help='Directory of data')
    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.path.join('.', 'visualization_mask'),
        help='Directory of visualization result')
    parser.add_argument(
        '--visualization_num',
        type=int,
        default=10,
        help='The image number of visualization')

    FLAGS, unparsed = parser.parse_known_args()
    main()
