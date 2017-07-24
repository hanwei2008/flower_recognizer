# -*- coding: utf-8 -*-
import glob
import os
import random
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import pickle

# 模型和样本路径的设置
# inception-V3瓶颈层节点个数
BOTTLENECK_TENSOR_SIZE = 2048
# 瓶颈层tenbsor name
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
# 图像输入tensor name
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

FLAGS = tf.app.flags.FLAGS


class DataUtils(object):
    '''
    完成数据处理相关工作
    '''

    def __init__(self, input_data, cache_dir, model_file):
        self.input_data = input_data
        if os.path.exists(FLAGS.f_category):
            self.dict_id_to_label, self.dict_label_to_id = pickle.load(open(FLAGS.f_category, 'rb'))
        else:
            list_labels = [f for f in os.listdir(input_data) if os.path.isdir(os.path.join(input_data, f))]
            dict_id_to_label = {i: label for i, label in enumerate(list_labels)}
            dict_label_to_id = {label: i for i, label in enumerate(list_labels)}
            pickle.dump([dict_id_to_label, dict_label_to_id], open(FLAGS.f_category, 'wb'))
            self.dict_id_to_label = dict_id_to_label
            self.dict_label_to_id = dict_label_to_id

        self.cache_dir = cache_dir
        # 读取已经训练好的Inception-v3模型。
        with gfile.FastGFile(model_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        self.bottleneck_tensor, self.jpeg_data_tensor = tf.import_graph_def(
            graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

    def id2label(self, target_id):
        label = self.dict_id_to_label[target_id]
        return label

    # 把样本中所有的图片列表并按训练、验证、测试数据分开
    def create_image_lists(self, testing_percentage, validation_percentage):
        result = {}
        sub_dirs = [x[0] for x in os.walk(self.input_data)]
        is_root_dir = True
        for sub_dir in sub_dirs:
            if is_root_dir:
                is_root_dir = False
                continue

            extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']

            file_list = []
            dir_name = os.path.basename(sub_dir)
            for extension in extensions:
                file_glob = os.path.join(self.input_data, dir_name, '*.' + extension)
                file_list.extend(glob.glob(file_glob))
            if not file_list: continue

            label_name = dir_name.lower()

            # 初始化
            training_images = []
            testing_images = []
            validation_images = []
            for file_name in file_list:
                base_name = os.path.basename(file_name)

                # 随机划分数据
                chance = np.random.randint(100)
                if chance < validation_percentage:
                    validation_images.append(base_name)
                elif chance < (testing_percentage + validation_percentage):
                    testing_images.append(base_name)
                else:
                    training_images.append(base_name)

            result[label_name] = {
                'dir': dir_name,
                'training': training_images,
                'testing': testing_images,
                'validation': validation_images,
            }
        return result

    # 函数通过类别名称、所属数据集和图片编号获取一张图片的地址
    def get_image_path(self, image_lists, image_dir, label_name, index, category):
        label_lists = image_lists[label_name]
        category_list = label_lists[category]
        mod_index = index % len(category_list)
        base_name = category_list[mod_index]
        sub_dir = label_lists['dir']
        full_path = os.path.join(image_dir, sub_dir, base_name)
        return full_path

    # 函数获取Inception-v3模型处理之后的特征向量的文件地址
    def get_bottleneck_path(self, image_lists, label_name, index, category):
        return self.get_image_path(image_lists, self.cache_dir, label_name, index, category) + '.txt'

    # 函数使用加载的训练好的Inception-v3模型处理一张图片，得到这个图片的特征向量。
    def run_bottleneck_on_image(self, sess, image_data):
        bottleneck_values = sess.run(self.bottleneck_tensor, {self.jpeg_data_tensor: image_data})

        bottleneck_values = np.squeeze(bottleneck_values)
        return bottleneck_values

    # 函数会先试图寻找已经计算且保存下来的特征向量，如果找不到则先计算这个特征向量，然后保存到文件
    def get_or_create_bottleneck(self, sess, image_lists, label_name, index, category):
        label_lists = image_lists[label_name]
        sub_dir = label_lists['dir']
        sub_dir_path = os.path.join(self.cache_dir, sub_dir)
        if not os.path.exists(sub_dir_path): os.makedirs(sub_dir_path)
        bottleneck_path = self.get_bottleneck_path(image_lists, label_name, index, category)
        if not os.path.exists(bottleneck_path):

            image_path = self.get_image_path(image_lists, self.input_data, label_name, index, category)

            image_data = gfile.FastGFile(image_path, 'rb').read()

            bottleneck_values = self.run_bottleneck_on_image(sess, image_data)

            bottleneck_string = ','.join(str(x) for x in bottleneck_values)
            with open(bottleneck_path, 'w') as bottleneck_file:
                bottleneck_file.write(bottleneck_string)
        else:

            with open(bottleneck_path, 'r') as bottleneck_file:
                bottleneck_string = bottleneck_file.read()
            bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

        return bottleneck_values

    # 函数随机获取一个batch的图片作为训练数据
    def get_random_cached_bottlenecks(self, sess, n_classes, image_lists, how_many, category):
        bottlenecks = []
        ground_truths = []
        for _ in range(how_many):
            label_index = random.randrange(n_classes)
            label_name = self.dict_id_to_label[label_index]
            image_index = random.randrange(65536)
            bottleneck = self.get_or_create_bottleneck(
                sess, image_lists, label_name, image_index, category)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)

        return bottlenecks, ground_truths

    # 获取全部的测试数据，并计算正确率
    def get_test_bottlenecks(self, sess, image_lists, n_classes):
        bottlenecks = []
        ground_truths = []
        for label_index, label_name in self.dict_id_to_label.iteritems():
            category = 'testing'
            for index, unused_base_name in enumerate(image_lists[label_name][category]):
                bottleneck = self.get_or_create_bottleneck(sess, image_lists, label_name, index, category)
                ground_truth = np.zeros(n_classes, dtype=np.float32)
                ground_truth[label_index] = 1.0
                bottlenecks.append(bottleneck)
                ground_truths.append(ground_truth)
        return bottlenecks, ground_truths
