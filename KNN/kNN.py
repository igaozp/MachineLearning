# -*-coding:utf-8-*-
"""
使用 kNN（k - 临近） 算法实现简易的手写数字识别
"""
import numpy as np
import operator
from os import listdir


def img_to_vector(filename):
    """
    将图片转换为向量
    :param filename: 图片的文件名
    :return: 图片相应的向量
    """
    vector = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lines = fr.readline()
        for j in range(32):
            vector[0, 32 * i + j] = int(lines[j])
    return vector


def classify0(x, data_set, labels, k):
    """
    kNN 算法
    :param x: 用于分类的向量
    :param data_set: 训练样本
    :param labels: 标签向量
    :param k: 选择最近邻居的数目
    :return: 识别的结果
    """
    data_set_size = data_set.shape[0]
    # 计算距离
    diff_matrix = np.tile(x, (data_set_size, 1)) - data_set
    sq_diff_matrix = diff_matrix ** 2
    sq_distances = sq_diff_matrix.sum(axis=1)
    distances = sq_distances ** 0.5
    sorted_distances_indicies = distances.argsort()
    class_count = {}
    # 选取距离最小的 k 个点
    for i in range(k):
        vote_label = labels[sorted_distances_indicies[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def handwriting_test():
    """
    测试手写数字的识别
    """
    hw_labels = []
    #  加载训练数据
    training_file_list = listdir('training_digits')
    m = len(training_file_list)
    training_matrix = np.zeros((m, 1024))
    for i in range(m):
        file_name_str = training_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        # 将图片对应的数字添加到标签列表
        hw_labels.append(class_num_str)
        # 将图片转为向量
        training_matrix[i, :] = img_to_vector('training_digits/%s' % file_name_str)

    # 加载测试数据
    test_file_list = listdir('test_digits')
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        file_name_str = test_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        # 将测试用的图片转化为向量
        vector_under_test = img_to_vector('test_digits/%s' % file_name_str)
        # 获取识别的结果
        classifier_result = classify0(vector_under_test, training_matrix, hw_labels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifier_result, class_num_str))
        # 统计错误数量
        if classifier_result != class_num_str:
            error_count += 1.0

    # 统计信息
    print("\n the total number of errors is: %d" % error_count)
    print("\n the total error rate is: %f" % (error_count / float(m_test)))
