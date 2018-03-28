from .layer import Layer

class Conv2D(Layer):
    counter = 0

    def __init__(self, line):
        self.name = 'conv2d_%d' % Conv2D.counter
        Conv2D.counter += 1

        # splitting by comma
        params = [x.strip() for x in line.split(',')]
        for param in params:
            oneParam = param.split("=")
            # print(oneParam)

    def summary(self):
        shape = "(%s, %s)" % (None, self.units)
        param = str(0)
        super(Conv2D, self).summary(shape, param)