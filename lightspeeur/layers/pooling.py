from tensorflow.keras import layers
from lightspeeur.drivers import Specification as Spec


class MaxPooling2D(layers.Layer):

    def __init__(self, name=None, **kwargs):
        super(MaxPooling2D, self).__init__(name=name, **kwargs)

    def call(self, inputs, *args, **kwargs):
        return layers.MaxPooling2D(pool_size=(Spec.POOLING_KERNEL, Spec.POOLING_KERNEL),
                                   strides=(Spec.POOLING_STRIDE, Spec.POOLING_STRIDE),
                                   padding=Spec.POOLING_PADDING)(inputs)

    def get_config(self):
        return super(MaxPooling2D, self).get_config()


class TopLeftPooling2D(layers.Layer):

    def __init__(self, name=None, **kwargs):
        super(TopLeftPooling2D, self).__init__(name=name, **kwargs)

    def call(self, inputs, *args, **kwargs):
        return inputs[:, ::Spec.POOLING_STRIDE, ::Spec.POOLING_STRIDE, :]

    def get_config(self):
        return super(TopLeftPooling2D, self).get_config()