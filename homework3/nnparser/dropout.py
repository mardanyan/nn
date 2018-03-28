from .layer import Layer
import collections

class Dropout(Layer):
    counter = 0

    def __initializer(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        # todo units is string, change to int
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.kernel_regularizer = kernel_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.input_spec = 2#InputSpec(min_ndim=2)
        self.supports_masking = True


        pass

    def __init__(self, line):
        self.name = "dense_%d" % Dense.counter
        Dense.counter += 1

        # define defaults
        defaultParams = collections.OrderedDict([("units", None),
                                                ("activation", None),
                                                ("use_bias", True),
                                                ("kernel_initializer", 'glorot_uniform'),
                                                ("bias_initializer", 'zeros'),
                                                ("kernel_regularizer", None),
                                                ("bias_regularizer", None),
                                                ("activity_regularizer", None),
                                                ("kernel_constraint", None),
                                                ("bias_constraint", None)])

        # splitting by comma
        params = [x.strip() for x in line.split(',')]
        index = 0
        for param in params:
            oneparam = param.split("=")
            if len(oneparam) is 2:
                # the case when given key/value pair, e.g. klor=tapak
                defaultParams[oneparam[0]] = oneparam[1]
                index = -1
            else:
                if index is -1:
                    raise Exception("Incorrect syntax")
                # case when given only param's value, e.g. 55
                key = list(defaultParams.keys())[index]
                defaultParams[key] = oneparam[0]
                index += 1

        #print(defaultParams)
        self.__initializer(**defaultParams)


    def summary(self):
        shape = "(%s, %s)" % (None, self.units)
        param = str(0)
        super(Dense, self).summary(shape, param)

