import tensorflow as tf

from tensorflow.keras import layers
from lightspeeur.drivers import conversion_initializer


class Dense(layers.Layer):
    def __init__(self,
                 units,
                 kernel_initializer=None,
                 bias_initializer=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(Dense, self).__init__(trainable=trainable, name=name, **kwargs)

        self.units = units
        self.kernel = None
        self.bias = None
        self.kernel_initializer = conversion_initializer(kernel_initializer, tf.initializers.glorot_uniform())
        self.bias_initializer = conversion_initializer(bias_initializer, tf.initializers.zeros())

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        last_dimension = int(input_shape[-1])

        self.input_spec = layers.InputSpec(min_ndim=2, axes={-1: last_dimension})
        self.kernel = self.add_weight('kernel',
                                      shape=(last_dimension, self.units),
                                      initializer=self.kernel_initializer,
                                      regularizer=None,
                                      constraint=None,
                                      trainable=True,
                                      dtype=self.dtype)
        self.bias = self.add_weight('bias',
                                    shape=(self.units,),
                                    initializer=self.bias_initializer,
                                    regularizer=None,
                                    constraint=None,
                                    trainable=True,
                                    dtype=self.dtype)
        self.built = True

    def call(self, inputs, *args, **kwargs):
        outputs = tf.matmul(inputs, self.kernel)
        outputs = tf.nn.bias_add(outputs, self.bias)
        return outputs

    def get_config(self):
        config = super(Dense, self).get_config()
        config.update({
            'units': self.units
        })
        return config
