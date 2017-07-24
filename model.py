# -*- coding: utf-8 -*-
import os

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

class PhotoRecognizer(object):
    '''
    基于google于2015年在ILSVRC比赛上发表的incept-v3模型进行迁移学习，完成其它图像识别任务
    '''

    def __init__(self, n_classes, data_processor, reuse=None):
        self.__n_classes = n_classes
        self.__data_processor = data_processor
        self.__reuse = reuse

    @property
    def reuse(self):
        return self.__reuse

    @reuse.setter
    def reuse(self, value):
        self.__reuse = value

    def model(self, bottleneck_input):
        # 定义一层全链接层
        with tf.variable_scope('final_training_ops', reuse=self.__reuse):
            weights = tf.Variable(tf.truncated_normal([FLAGS.bottleneck_tensor_size, self.__n_classes], stddev=0.001))
            biases = tf.Variable(tf.zeros([self.__n_classes]))
            logits = tf.matmul(bottleneck_input, weights) + biases
            final_tensor = tf.nn.softmax(logits)
            predictions = tf.argmax(final_tensor, 1)
        return predictions, logits
