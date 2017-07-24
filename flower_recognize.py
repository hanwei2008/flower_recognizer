# -*- coding: utf-8 -*-
import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from data_utils import DataUtils
from model import PhotoRecognizer

'''
todo:增加predict
'''

# 模型和样本路径的设置
# inception-V3瓶颈层节点个数
BOTTLENECK_TENSOR_SIZE = 2048
# 瓶颈层tenbsor name
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
# 图像输入tensor name
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

# v3 modefile
MODEL_FILE = 'data/flower/model/inception_dec_2015/tensorflow_inception_graph.pb'

# 特征向量 save path
CACHE_DIR = 'data/flower/model/bottleneck'
# 数据path
INPUT_DATA = 'data/flower/data/flower_photos'

# 验证数据 percentage
VALIDATION_PERCENTAGE = 10
# 测试数据 percentage
TEST_PERCENTAGE = 10

# 神经网络参数的设置
LEARNING_RATE = 0.01
STEPS = 400
BATCH = 100

tf.app.flags.DEFINE_integer('bottleneck_tensor_size', 2048, 'bottleneck tensor size')
# 瓶颈层tenbsor name
tf.app.flags.DEFINE_string('bottleneck_tensor_name', 'pool_3/_reshape:0', '瓶颈层tenbsor name')
# 图像输入tensor name
tf.app.flags.DEFINE_string('jpeg_data_tensor_name', 'DecodeJpeg/contents:0', '图像输入tensor name')

# v3 modefile
tf.app.flags.DEFINE_string('model_file', 'data/flower/model/inception_dec_2015/tensorflow_inception_graph.pb',
                           'v3 modefile')

# 特征向量 save path
tf.app.flags.DEFINE_string('cache_dir', 'data/flower/model/bottleneck', '特征向量 save path')
# 数据path
tf.app.flags.DEFINE_string('input_data', 'data/flower/data/flower_photos', '数据path')
tf.app.flags.DEFINE_string('f_category', 'data/flower/data/category.txt', 'category map file')
tf.app.flags.DEFINE_integer('n_classes', 5, 'number of classes')
tf.app.flags.DEFINE_boolean('predict', False, 'is predict mode')
tf.app.flags.DEFINE_string('input_image', None, 'input image path whereas predict')
tf.app.flags.DEFINE_string('checkpoint_path', 'data/flower/model/checkpoints/', 'the checkpoint path')
FLAGS = tf.app.flags.FLAGS


def train(data_processor, photo_recognizer):
    image_lists = data_processor.create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    n_classes = FLAGS.n_classes
    # 定义新的神经网络输入
    bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
    ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')

    predicts, logits = photo_recognizer.model(bottleneck_input)

    # 定义交叉熵损失函数。
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 定义训练步骤
    global_step = tf.Variable(0, trainable=False)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean, global_step=global_step)

    # 计算正确率。
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(predicts, tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        # 训练过程。
        last_validation_accuracy = -1
        for i in range(STEPS):
            train_bottlenecks, train_ground_truth = data_processor.get_random_cached_bottlenecks(
                sess, n_classes, image_lists, BATCH, 'training')
            sess.run(train_step,
                     feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})

            if i % 100 == 0 or i + 1 == STEPS:
                validation_bottlenecks, validation_ground_truth = data_processor.get_random_cached_bottlenecks(
                    sess, n_classes, image_lists, BATCH, 'validation')
                validation_accuracy = sess.run(evaluation_step, feed_dict={
                    bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                print('Step %d: Validation accuracy on random sampled %d examples = %.1f%%' %
                      (i, BATCH, validation_accuracy * 100))
                if validation_accuracy > last_validation_accuracy:
                    last_validation_accuracy = validation_accuracy
                    saver.save(sess, FLAGS.checkpoint_path, global_step=global_step)

        # 在最后的测试数据上测试正确率。
        test_bottlenecks, test_ground_truth = data_processor.get_test_bottlenecks(
            sess, image_lists, n_classes)
        test_accuracy = sess.run(evaluation_step, feed_dict={
            bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

        # save model
        saver.save(sess, FLAGS.checkpoint_path, global_step=global_step)


def predict(image_path, data_processor, photo_recognizer):
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    with tf.Session() as sess:
        bottleneck = data_processor.run_bottleneck_on_image(sess, image_data)
        bottleneck = np.squeeze(bottleneck)
        bottleneck = [bottleneck]

        # 定义新的神经网络输入
        bottleneck_input = tf.placeholder(tf.float32, [None, FLAGS.bottleneck_tensor_size],
                                          name='BottleneckInputPlaceholder')
        prediction = photo_recognizer.model(bottleneck_input)
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            return None
        prediction, _ = sess.run(prediction, feed_dict={bottleneck_input: bottleneck})

    prediction_label = data_processor.id2label(prediction[0])
    return prediction_label


def main():
    data_processor = DataUtils(FLAGS.input_data, FLAGS.cache_dir, FLAGS.model_file)
    photo_recognizer = PhotoRecognizer(FLAGS.n_classes, data_processor)
    if FLAGS.predict:
        if FLAGS.input_image is not None:
            predict_result = predict(FLAGS.input_image, data_processor, photo_recognizer)
            print('图片%s的类别为%s' % (FLAGS.input_image, predict_result))
    else:
        train(data_processor, photo_recognizer)


if __name__ == '__main__':
    main()
