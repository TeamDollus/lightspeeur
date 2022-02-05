import tensorflow as tf

from tensorflow.keras import layers
from lightspeeur.drivers import permute_axis
from lightspeeur.drivers import Specification

EPSILON = 1e-6


def round_as_chip(x):
    y = tf.where(x >= 0, tf.math.floor(x + 0.5), tf.math.ceil(x - 0.5))
    return x + tf.stop_gradient(y - x)


def quantized_shift(x, shift):
    y = round_as_chip(x * (2 ** shift)) / (2 ** shift)
    return x + tf.stop_gradient(y - x)


def quantized_bit_shift(x, bit_max):
    steps = (2.0 ** (bit_max - 1) - 1) / tf.reduce_max(tf.abs(x))
    shift = tf.cast(tf.math.log(steps) / tf.math.log(2.0), tf.int32)
    shift = tf.cast(shift, tf.float32)
    return tf.stop_gradient(shift)


def compute_quantized_shift(specification: Specification, kernel, biases, bit_mask):
    bit_weight, bit_bias = specification.quantization_scheme(bit_mask)
    shifted_weight = quantized_bit_shift(kernel, bit_weight)
    if biases is None:
        shifted = tf.clip_by_value(shifted_weight, Specification.MIN_SHIFT, Specification.MAX_SHIFT)
        return tf.stop_gradient(shifted)
    shifted_bias = quantized_bit_shift(biases, bit_bias)
    shifted = tf.clip_by_value(tf.minimum(shifted_weight, shifted_bias),
                               Specification.MIN_SHIFT, Specification.MAX_SHIFT)
    return tf.stop_gradient(shifted)


def quantize_kernel(x, shift, bits):
    transposed_x = tf.transpose(x, perm=permute_axis('HWIO', 'OIHW'))
    abs_transposed_x = tf.abs(transposed_x)
    if bits == 4 or bits > 31:
        raise ValueError('Invalid quantization scheme for {} bits'.format(bits))

    # these variables will be used for bits==3 or bits==5
    coefficients = None
    steps = None

    if bits > 5:
        value = transposed_x
    elif bits == 5:
        value = tf.reduce_max(abs_transposed_x, axis=(2, 3), keepdims=True)
        steps = tf.where(value == 0, tf.ones(tf.shape(value)) * EPSILON, value) / 15.0
        coefficients = round_as_chip(transposed_x / steps)
        value = steps
    else:
        value = tf.reduce_mean(abs_transposed_x, axis=(2, 3), keepdims=True)
        if bits == 3:
            steps = tf.where(value == 0, tf.ones(tf.shape(value)) * EPSILON, value) / 4.0
            coefficients = tf.cast(tf.cast(abs_transposed_x / steps, tf.int32), tf.float32)
            value = steps

    value = quantized_shift(value, shift)
    if bits == 1:
        y = tf.where(transposed_x >= 0,
                     tf.ones(tf.shape(transposed_x)) * value,
                     tf.ones(tf.shape(transposed_x)) * value * -1)
    elif bits == 2:
        y = tf.where(abs_transposed_x >= value / 4,
                     tf.ones(tf.shape(transposed_x)) * value,
                     tf.zeros(tf.shape(transposed_x)))
        y = tf.where(transposed_x >= 0, y, y * -1)
    elif bits == 3:
        y = tf.where(coefficients >= 3, tf.fill(tf.shape(coefficients), 4.0), coefficients)
        y = tf.where(transposed_x >= 0, y * steps, -1 * y * steps)
    elif bits == 5:
        y = coefficients * value
    else:
        y = value

    y = tf.transpose(y, perm=permute_axis('OIHW', 'HWIO'))
    return x + tf.stop_gradient(y - x)


class QuantizableLayer(layers.Layer):

    def __init__(self, quantize=False, **kwargs):
        super(QuantizableLayer, self).__init__(**kwargs)

        self.quantize = quantize
