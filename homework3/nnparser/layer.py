
class Layer:
    def __init__(self):
        pass

    def summary(self, shape, param):
        print("_" * 65)
        str = "%s (%s)" % (self.name, type(self).__name__)
        pos = 0 if (len(str)) > 28 else 28 - len(str)
        str += " " * pos + shape
        pos = 0 if (len(str)) > 52 else 52 - len(str)
        str += " " * pos + param
        print(str)