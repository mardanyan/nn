from keras.datasets import mnist
import keras
import numpy as np
from matplotlib import pyplot as plt

def showImage(img):


class nn:
    def __init__(self):
        (self.x_train, self.y_train), (self.y_test, self.y_test) = mnist.load_data()
        self.x_train = self.x_train.reshape(60000, 784)
        self.w = np.random.randn(10, 784)
        self.alpha = 0.001


    def showImage(self, img):
        plt.imshow(img)
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

    def loss(self, softMax, index):
        return -np.log(softMax, self.y_train[index])

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
            lossValue = self.loss(softMax, index)

            # loss's ajancjal @st w-i
            gradientValue = self.grad(index, softMax)

            self.update(gradientValue)

    def run(self):
        self.epoche()

    def printResult(self):
        print()


if __name__ == "__main__":
    n = nn()
    n.run()










