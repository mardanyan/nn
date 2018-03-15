#!/usr/bin/evn python

'''MNIST exampl on other dataset
MNIST example - https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py
FASHION-MNIST - https://github.com/zalandoresearch/fashion-mnist
'''

from __future__ import print_function

import keras
import numpy as np
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

from matplotlib import pyplot as plt


is_fashion = True;
batch_size = 128
num_classes = 10
epochs = 3

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#(x_train, y_train), (x_test, y_test) = mnist.load_data()


plt.show(x_train[0])

print(x_train.shape)
print(y_train.shape)

# print(x_test.shape)
# print(y_test.shape)

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)


# print(x_train.shape)
# print(y_train.shape)

# print(x_test.shape)
# print(y_test.shape)

# print(type(x_test[0, 0]))

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

x_train /= 255
x_test /= 255

# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

# print(y_train.shape)


# convert class vectors to binary class metrics
y_train_bac = y_train
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# print(y_train_bac.shape)
# print(y_train.shape)
#
# print(y_train_bac[0])
# print(y_train[0])
# print(y_train_bac[1])
# print(y_train[1])
# print(y_train_bac[2])
# print(y_train[2])



model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.summary()

print(len(model.layers))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# print(y_train[:10])



