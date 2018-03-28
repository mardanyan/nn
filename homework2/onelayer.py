from keras.datasets import mnist
import keras
import numpy as np
from matplotlib import pyplot as plt


class NeuralNetwork:
    def __init__(self):
        # Assign random weights to a 3 x 1 matrix,
        self.synaptic_weights = 2 * np.random.random((10, 784)) - 1
        # self.w = np.random.randn(10, 784)

        self.alpha = 0.001
        self.epoch = 100

    def showImage(self, index):
        plt.imshow(self.x_train_orig[index])
        plt.colorbar()
        # plt.title('plot for image ")
        plt.show()

    # The Sigmoid function
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(d):
        d = d - np.max(d)
        exp = np.exp(d)
        summ = np.sum(exp)
        return exp / summ

    def dense(self, index):
        return self.synaptic_weights.dot(self.x_train[index])

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
        self.synaptic_weights = self.synaptic_weights - self.alpha * gradient.T

    def epoche(self):

        for index in range(600):
            denseValue = self.dense(index)

            # calc softmax
            softMax = NeuralNetwork.softmax(denseValue)

            # calc loss
            # lossValue = self.loss(index)

            # loss's ajancjal @st w-i
            gradientValue = self.gradient(index, softMax)

            self.update(gradientValue)

    def train(self, x_train, y_train, x_test, y_test):
        self.x_train_orig = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.x_train = self.x_train_orig.reshape(60000, 784)
        self.x_test = self.x_test.reshape(10000, 784)
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255

        print("x_train shape %s" % (self.x_train_orig.shape,))
        print("x_test shape %s" % (self.x_test.shape,))

        print("w shape %s" % (self.synaptic_weights.shape,))
        print("w row0 %s" % self.synaptic_weights[0, 0:10])

        for iteration in range(self.epoch):
            # Pass the training set through the network.
            output = self.learn(self.x_train)

            # Calculate the error
            # error = outputs - output

    # The neural network thinks.
    def learn(self, inputs):
        return self.__sigmoid(np.dot(inputs, self.synaptic_weights))

        # self.epoche()

    def printResult(self):
        print()


if __name__ == "__main__":
    nn = NeuralNetwork()

    # n.showImage(1)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    nn.train(x_train, y_train, x_test, y_test)










