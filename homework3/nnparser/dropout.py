from .layer import Layer
import collections

class Dropout(Layer):
    counter = 1

    def __init__(self, line):
        self.name = "dropout_%d" % Dropout.counter
        Dropout.counter += 1

    def summary(self):
        shape = "(%s, %s)" % (None, self.units)
        param = str(0)
        super(Dropout, self).summary(shape, param)
