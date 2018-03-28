
class Layer:
    def __init__(self, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        # batch_size = None
        # self.batch_input_shape = (batch_size,) + tuple(kwargs['input_shape'])


    def summary(self, shape, param):
        print("_" * 65)
        str = "%s (%s)" % (self.name, type(self).__name__)
        pos = 0 if (len(str)) > 28 else 28 - len(str)
        str += " " * pos + shape
        pos = 0 if (len(str)) > 52 else 52 - len(str)
        str += " " * pos + param
        print(str)

    def getParams(self):
        return 0

    def getTrainableParams(self):
        return 0