#### Libraries
# Standard library
import pickle
import gzip
# Third-party libraries
import numpy as np

def load_data():
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f,encoding='bytes')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """
    通过load_data()得到的是一个三元组列表(training_data, validation_data,test_data)
    training_data 是一个50000大小的二维元组（x，y），x是一个784维的向量，因为每一张图片
    都是28*28大小的，y是一个10维的向量，因为数字是0~9之间。
    ``validation_data`` 和``test_data``是一个包含10000张图片的2元组，因为在load_data()
    中并没有对其进行什么处理，所以在这里把这些图片进行向量化，方便使用。
    """
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """
    把y进行向量化
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

