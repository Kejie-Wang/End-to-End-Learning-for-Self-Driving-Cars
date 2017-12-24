import argparse
import os

import tensorflow as tf

from model import Nivdia_Model
import reader

FLAGS = None


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


def train():
    x_image = tf.placeholder(tf.float32, [None, 66, 200, 3])
    y = tf.placeholder(tf.float32, [None, 1])
    keep_prob = tf.placeholder(tf.float32)

    model = Nivdia_Model(x_image, y, keep_prob, FLAGS)

    # dataset reader
    dataset = reader.Reader(FLAGS.data_dir, FLAGS)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train',
                                             sess.graph)
        # initialize all varibales
        sess.run(tf.global_variables_initializer())

        min_validation_loss = float('Inf')
        # restore model
        if not FLAGS.disable_restore:
            path = tf.train.latest_checkpoint(FLAGS.model_dir)
            if not (path is None):
                saver.restore(sess, path)
                # validation
                min_validation_loss = batch_eval(
                    model.loss, dataset.validation, x_image, y, keep_prob,
                    FLAGS.batch_size, sess)
                print('Restore model from', path)

        for i in range(FLAGS.max_steps):
            batch_x, batch_y = dataset.train.next_batch(FLAGS.batch_size)
            # train model
            if i % 100 == 99:  # Record execution stats
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run(
                    [merged, model.optimization],
                    feed_dict={x_image: batch_x,
                               y: batch_y,
                               keep_prob: 0.8},
                    options=run_options,
                    run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            else:
                summary, _ = sess.run(
                    [merged, model.optimization],
                    feed_dict={
                        x_image: batch_x,
                        y: batch_y,
                        keep_prob: 0.8
                    })
            train_writer.add_summary(summary, i)

            # validation
            validation_loss = batch_eval(model.loss, dataset.validation,
                                         x_image, y, keep_prob,
                                         FLAGS.batch_size, sess)
            if (validation_loss < min_validation_loss):
                min_validation_loss = validation_loss
                saver.save(sess, os.path.join(FLAGS.model_dir, "model.ckpt"))

            if i % FLAGS.print_steps == 0:
                loss = sess.run(
                    model.loss,
                    feed_dict={
                        x_image: batch_x,
                        y: batch_y,
                        keep_prob: 1.0
                    })
                print("Step", i, "train_loss: ", loss, "validation_loss: ",
                      validation_loss)

        train_writer.close()


def main():
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    if not tf.gfile.Exists(FLAGS.model_dir):
        tf.gfile.MakeDirs(FLAGS.model_dir)
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--max_steps',
        type=int,
        default=20000,
        help='Number of steps to run trainer')
    parser.add_argument(
        '--print_steps',
        type=int,
        default=100,
        help='Number of steps to print training loss')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Initial learning rate')
    parser.add_argument(
        '--batch_size', type=int, default=500, help='Train batch size')
    parser.add_argument(
        '--data_dir',
        type=str,
        default=os.path.join('.', 'driving_dataset'),
        help='Directory of data')
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join('.', 'logs'),
        help='Directory of log')
    parser.add_argument(
        '--model_dir',
        type=str,
        default=os.path.join('.', 'saved_model'),
        help='Directory of saved model')
    parser.add_argument(
        '--disable_restore',
        type=int,
        default=0,
        help='Whether disable restore model from model directory')

    FLAGS, unparsed = parser.parse_known_args()
    main()
