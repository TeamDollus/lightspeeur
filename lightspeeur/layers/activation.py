import numpy as np
import tensorflow as tf

from lightspeeur.drivers import Specification
from lightspeeur.layers.quantization import round_as_chip, QuantizableLayer


class ReLU(QuantizableLayer):
    def __init__(self,
                 chip_id,
                 quantize=False,
                 cap=None,
                 activation_bits=5,
                 trainable=False,
                 name=None,
                 **kwargs):

        super(ReLU, self).__init__(quantize=quantize, trainable=trainable, name=name, **kwargs)

        self.chip_id = chip_id
        self.specification = Specification(chip_id)
        self.cap = self.specification.max_activation() if cap is None else cap
        self.activation_bits = activation_bits
        self.relu_output = None
        self.relu_cap = None

    def build(self, input_shape):
        regularizer = tf.keras.regularizers.l2(l=0.01) if self.trainable else None
        self.relu_cap = self.add_weight('relu_cap',
                                        shape=(1,),
                                        initializer=tf.keras.initializers.Constant(np.array([self.cap],
                                                                                            dtype=np.float32)),
                                        regularizer=regularizer,
                                        constraint=None,
                                        trainable=self.trainable,
                                        dtype=self.dtype)
        self.built = True

    def call(self, inputs, *args, **kwargs):
        outputs = inputs
        if self.quantize:
            max_activation = self.specification.max_activation(self.activation_bits)
            outputs = 0.5 * (tf.abs(outputs) - tf.abs(outputs - self.relu_cap) + self.relu_cap)
            outputs = round_as_chip(outputs * max_activation / self.relu_cap) * self.relu_cap / max_activation
        else:
            outputs = tf.nn.relu(outputs)
        return outputs

    def get_config(self):
        config = super(ReLU, self).get_config()
        config.update({
            'chip_id': self.chip_id,
            'quantize': self.quantize,
            'cap': self.cap,
            'activation_bits': self.activation_bits
        })
        return config
