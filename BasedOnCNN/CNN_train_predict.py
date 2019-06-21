from tensorflow.examples.tutorials.mnist import input_data
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation
from keras.optimizers import Adam
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np

import math
import random

model = Sequential()
class CNN():

    def getDate(self):
        mnist = input_data.read_data_sets('MINST_data', one_hot=True)
        x_train = mnist.train.images
        x_train = x_train.reshape(-1, 1, 28, 28)
        y_train = mnist.train.labels
        x_test = mnist.test.images
        x_test = x_test.reshape(-1, 1, 28, 28)
        y_test = mnist.test.labels
        return x_train, y_train, x_test, y_test

    def train(self):
        x_train, y_train, x_test, y_test = self.getDate()

        model.add(Conv2D(input_shape=(1, 28, 28), filters=32, kernel_size=(5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        model.add(Conv2D(64,(5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2), padding='same'))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))

        model.add(Dense(10))
        model.add(Activation('softmax'))

        model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

        print("Train---------")
        model.fit(x_train, y_train, batch_size=32, epochs=1)

        print("Test---------")
        loss, accuracy = model.evaluate(x_test, y_test)
        print(loss, accuracy)
    def predict(self, x, y):
        def drawDigit3(position, image, title, isTrue):
            plt.subplot(*position)
            plt.imshow(image.reshape(-1, 28), cmap='gray_r')
            plt.axis('off')
            if not isTrue:
                plt.title(title, color='red')
            else:
                plt.title(title)

        def batchDraw3(test_X, test_y):
            selected_index = random.sample(range(len(test_y)), k=1) # 从range(len(test_y))范围内随机生成1个数
            images = test_X[selected_index]
            labels = test_y[selected_index]
            predict_labels = model.predict(images)
            image_number = images.shape[0]
            row_number = math.ceil(image_number ** 0.5) # math.ceil（x）对x进行向上取整
            column_number = row_number
            plt.figure(figsize=(row_number + 4, column_number + 4)) # 调整所显示图片的大小
            for i in range(row_number):
                for j in range(column_number):
                    index = i * column_number + j
                    if index < image_number:
                        position = (row_number, column_number, index + 1)
                        image = images[index]
                        actual = np.argmax(labels[index])
                        predict = np.argmax(predict_labels[index])
                        isTrue = actual == predict
                        title = 'actual:%d\n predict:%d' % (actual, predict)
                        drawDigit3(position, image, title, isTrue)

        batchDraw3(x, y)
        plt.show()

if __name__ == "__main__":
    o = CNN()
    o.train()
    x_train, y_train, x_test, y_test = o.getDate()
    # print("Truth label is :", y_test[100])
    o.predict(x_test, y_test)