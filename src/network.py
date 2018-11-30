"""
network.py
~~~~~~~~~~
这是一个前馈神经网络，实现了随机梯度下降的学习算法。梯度是通过反向传播计算出来的。

"""


import random


import numpy as np

class Network(object):

    def __init__(self, sizes):
        """列表sizes包括了神经元数量，和层数。例如 sizes=[2,3,1]则表示有三层网络结构，第一层有2个
        神经元，第二层有3个神经元，第三层有1个神经元。权重和偏移值是通过高斯分布随机初始化的（0或1）。
        第一层是输入层，且不设置偏移值，我们不会为这些神经元设置任何偏移值，因为偏移值仅用于计算后续层的输出
        """

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """如果输入的是a，则返回一个输出结果，即是经过激活函数的输出，
        通常的激活函数有：sigmoid  relu  tanh。
        但sigmoid函数会出现梯度消失问题，所以常见的都是采用relu函数
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """训练神经网络使用的是随机梯度下降法进行训练，train_data 是一个二元组列表，如（x，y）
        x是特征，y是观测值。epochs是执行周期，mini_batch_size是minist数据集一批的大小，eta是
        学习率，随机梯度下降法：w = w - eta * p  [注：p表示偏导数]
        如果提供了``test_data``，那么将在每个周期之后针对测试数据评估网络模型，并打印出部分进度。
        这对于跟踪进度很有用，但会大大减慢速度。在这个方法中主要的还只是做了一些数据的处理问题，比如
        因为mini_batch_size表示的是一批的大小，所以便可以通过：
        mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
        实现将数据将数据分成多少批。
        """
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """在这一部分通过反向传播实现神经网络的权重和偏移值的更新，mini_batch是二元组数据列表（x，y），
        eta是学习率。
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)] # 此处采用梯度下降更新权重
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)] # 采用梯度下降更新偏移值

    def backprop(self, x, y):
        """
        参数x是一个输入值，y是一个观测值。返回表示损失函数C_x的梯度元组(nabla_b, nabla_w)，
        nabla_b, nabla_w 是层与层之间的numpy列表类似与权重和偏移值，并且在此全部初始化为0。
        从最后一层往前依次求取偏w和偏b的导数
        注：一个小小技巧，该层求取w偏导=往后一层偏导 * 该层的输出
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # 前馈
        activation = x
        activations = [x] # 存储了所有层与层之间的激活值
        zs = [] # 列表存储了所有层与层之间的z向量,即 z=w * x + b
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])  # 此处是对损失函数求输出层偏移值的偏导
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())  # 此处是对损失函数求输出层权重的偏导
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp   #  对偏移值求偏导数
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())  # 对权重求偏导数
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        返回神经网络输出层认为是正确的那个值，这个值是被激活函数处理后最大的那个值
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        返回损失函数C_x的导数,output_activations是输出的值, y观测值
        """
        return (output_activations-y)


def sigmoid(z):
    """sigmoid函数."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """sigmoid函数的导数."""
    return sigmoid(z)*(1-sigmoid(z))
