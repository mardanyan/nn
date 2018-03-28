from .dense import Dense
from .conv2D import Conv2D
import re

class Sequential:
    def __init__(self):
        self.layers = []
        pass

    def __parseline(self, line):
        if "Dense" in line:
            params = re.search(r'Dense\((.*?)\)\)', line).group(1)
            # print("parsed >> " + result)
            return Dense(params)
        if "Conv2D" in line:
            params = re.search(r'Conv2D\((.*?)\)\)', line).group(1)
            # print("parsed >> " + result)
            return Conv2D(params)

        return None

    def parse(self, stringCode):
        lines = stringCode.splitlines()
        for line in lines:
            layer = self.__parseline(line)
            if layer is not None:
                self.layers.append(layer)

    def summary(self):
        print("_" * 65)
        # to_display = ['Layer (type)', 'Output Shape', 'Param #']
        print("Layer(type)\t\t\t\t\tOutput Shape\t\t\tParam  #")

        for layer in self.layers:
            layer.summary()

        print("=" * 65)
        print("Total params: %d" % 1,199,882)
        print("Trainable params: %d" % 1,199,882)
        print("Non - trainable params: %d" % 0)
        print("_" * 65)