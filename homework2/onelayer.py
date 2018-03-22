from keras.datasets import mnist
import keras
import numpy as np
from matplotlib import pyplot as plt


class nn:
    def __init__(self):
        (self.x_train_orig, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

        print("x_train shape %s" % (self.x_train_orig.shape,))
        print("x_test shape %s" % (self.x_test.shape,))

        self.x_train = self.x_train_orig.reshape(60000, 784)
        self.x_test = self.x_test.reshape(10000, 784)
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255

        self.w = np.random.randn(10, 784)
        print("w shape %s" % (self.w.shape,))
        print("w row0 %s" % self.w[0, 0:10])
        self.alpha = 0.001

    def showImage(self, index):
        plt.imshow(self.x_train_orig[index])
        plt.colorbar()
        # plt.title('plot for image ")
        plt.show()

    def softmax(d):
        d = d - np.max(d)
        exp = np.exp(d)
        summ = np.sum(exp)
        return exp / summ

    def dense(self, index):
        return self.w.dot(self.x_train[index])

    # def loss(self, index):
    #     return -np.log(self.y_train[index])

    def gradient(self, index, softMax):
        # los-i acancjal@ @st Y~
        # grad = Y~ - xT
        onehot = keras.utils.to_categorical(self.y_train[index], 10)
        yaliq = -(onehot - softMax)
        # print(yaliq.shape)
        # verevin@ acancjalner @st wx
        # hima acancjal@ @st w
        # print(x.shape)
        return np.outer(self.x_train[index], yaliq)

    def update(self, gradient):
        self.w = self.w - self.alpha * gradient.T

    def epoche(self):

        for index in range(600):
            denseValue = self.dense(index)

            # calc softmax
            softMax = nn.softmax(denseValue)

            # calc loss
            # lossValue = self.loss(index)

            # loss's ajancjal @st w-i
            gradientValue = self.gradient(index, softMax)

            self.update(gradientValue)

    def run(self):
        self.epoche()

    def printResult(self):
        print()


if __name__ == "__main__":
    n = nn()
    # n.showImage(1)
    n.run()










