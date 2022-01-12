import tensorflow as tf

from tensorflow.keras import layers
from lightspeeur.drivers import Specification, UpSampleFillingMode
from lightspeeur.drivers import conversion_initializer
from lightspeeur.layers.quantization import compute_quantized_shift, quantize_kernel, quantized_shift, QuantizableLayer
from lightspeeur.layers.constraints import ClippingBiasConstraint


def pad_for_chip(inputs, strides, kernel_size):
    if kernel_size == (1, 1):
        return inputs, 'VALID'
    elif kernel_size == (3, 3) and (strides == 2 or strides == (2, 2)):
        # batch_size, height, width, channel
        inputs = tf.pad(inputs, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]])  # pad top and left
        return inputs, 'VALID'
    else:
        return inputs, 'SAME'


def quantized_conv2d(specification, inputs, kernel, biases, bit_mask, strides, padding):
    shifts = compute_quantized_shift(specification,
                                     kernel=kernel,
                                     biases=biases,
                                     bit_mask=bit_mask)
    kernel = quantize_kernel(kernel, shifts, bit_mask)
    outputs = tf.nn.conv2d(inputs, kernel, strides, padding)
    if biases is not None:
        biases = quantized_shift(biases, shifts)
        outputs = tf.nn.bias_add(outputs, biases)
    return outputs


def quantized_depthwise_conv2d(specification, inputs, depthwise_kernel, biases, bit_mask, strides, padding):
    shifts = compute_quantized_shift(specification,
                                     kernel=depthwise_kernel,
                                     biases=biases,
                                     bit_mask=bit_mask)
    depthwise_kernel = quantize_kernel(depthwise_kernel, shifts, bit_mask)
    outputs = tf.nn.depthwise_conv2d(inputs, depthwise_kernel, strides, padding)
    if biases is not None:
        biases = quantized_shift(biases, shifts)
        outputs = tf.nn.bias_add(outputs, biases)
    return outputs


class DepthwiseConv2D(QuantizableLayer):

    def __init__(self,
                 chip_id,
                 kernel_size=Specification.DEFAULT_CONV_KERNEL_SIZE,
                 strides=(1, 1),
                 depth_multiplier=1,
                 use_bias=True,
                 quantize=False,
                 bit_mask=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 clip_bias=False,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(DepthwiseConv2D, self).__init__(quantize=quantize, trainable=trainable, name=name, **kwargs)

        if strides not in Specification.ALLOWED_CONV_STRIDES:
            raise ValueError('Convolution stride {} is not supported'.format(strides))

        self.depthwise_kernel = None
        self.bias = None
        self.kernel_initializer = conversion_initializer(kernel_initializer, tf.initializers.glorot_uniform())
        self.bias_initializer = conversion_initializer(bias_initializer, tf.initializers.zeros())
        self.kernel_size = kernel_size if type(kernel_size) is tuple else tuple(kernel_size)
        self.strides = strides if type(strides) is tuple else tuple(strides)
        self.depth_multiplier = depth_multiplier
        self.use_bias = use_bias
        self.clip_bias = clip_bias
        self.bit_mask = bit_mask
        self.chip_id = chip_id
        self.specification = Specification(self.chip_id)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_channel = int(input_shape[-1])
        # (kernel height, kernel width, input channels, depth multiplier)
        kernel_shape = self.kernel_size + (input_channel, self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(name='depthwise_kernel',
                                                shape=kernel_shape,
                                                initializer=self.kernel_initializer,
                                                regularizer=None,
                                                constraint=None,
                                                trainable=True,
                                                dtype=self.dtype)
        if self.use_bias:
            if self.clip_bias:
                constraint = ClippingBiasConstraint(self.depthwise_kernel)
            else:
                constraint = None
            self.bias = self.add_weight(name='bias',
                                        shape=(input_channel * self.depth_multiplier,),
                                        initializer=self.bias_initializer,
                                        regularizer=None,
                                        constraint=constraint,
                                        trainable=True,
                                        dtype=self.dtype)
        else:
            self.bias = None
        self.input_spec = layers.InputSpec(min_ndim=4,
                                           axes={-1: input_channel})
        self.built = True

    def call(self, inputs, *args, **kwargs):
        inputs, padding = pad_for_chip(inputs, self.strides, self.kernel_size)
        strides = (1,) + self.strides + (1,)
        if self.quantize:
            outputs = quantized_depthwise_conv2d(self.specification,
                                                 inputs=inputs,
                                                 depthwise_kernel=self.depthwise_kernel,
                                                 biases=self.bias,
                                                 bit_mask=self.bit_mask,
                                                 strides=strides,
                                                 padding=padding)
        else:
            outputs = tf.nn.depthwise_conv2d(inputs, self.depthwise_kernel, strides, padding)
            if self.use_bias:
                outputs = tf.nn.bias_add(outputs, self.bias)

        return outputs

    def get_config(self):
        config = super(DepthwiseConv2D, self).get_config()
        config.update({
            'chip_id': self.chip_id,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'depth_multiplier': self.depth_multiplier,
            'use_bias': self.use_bias,
            'quantize': self.quantize,
            'bit_mask': self.bit_mask,
            'clip_bias': self.clip_bias
        })
        return config


class Conv2D(QuantizableLayer):

    def __init__(self,
                 filters,
                 kernel_size,
                 chip_id,
                 strides=(1, 1),
                 use_bias=True,
                 quantize=False,
                 bit_mask=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 clip_bias=False,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(Conv2D, self).__init__(quantize=quantize, trainable=trainable, name=name, **kwargs)

        if kernel_size not in Specification.ALLOWED_CONV_KERNEL_SIZES:
            raise ValueError("Kernel {} is not supported".format(kernel_size))
        if strides not in Specification.ALLOWED_CONV_STRIDES:
            raise ValueError('Convolution stride {} is not supported'.format(strides))

        self.kernel = None
        self.bias = None
        self.kernel_initializer = conversion_initializer(kernel_initializer, tf.initializers.glorot_uniform())
        self.bias_initializer = conversion_initializer(bias_initializer, tf.initializers.zeros())
        self.filters = filters
        self.kernel_size = kernel_size if type(kernel_size) is tuple else tuple(kernel_size)
        self.strides = strides if type(strides) is tuple else tuple(strides)
        self.use_bias = use_bias
        self.clip_bias = clip_bias
        self.bit_mask = bit_mask
        self.chip_id = chip_id
        self.specification = Specification(self.chip_id)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_channel = int(input_shape[-1])
        # (kernel height, kernel width, input channels, output channels)
        kernel_shape = self.kernel_size + (input_channel, self.filters)

        self.kernel = self.add_weight(name='kernel',
                                      shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=None,
                                      constraint=None,
                                      trainable=True,
                                      dtype=self.dtype)
        if self.use_bias:
            if self.clip_bias:
                constraint = ClippingBiasConstraint(self.kernel)
            else:
                constraint = None
            self.bias = self.add_weight(name='bias',
                                        shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        regularizer=None,
                                        constraint=constraint,
                                        trainable=True,
                                        dtype=self.dtype)
        else:
            self.bias = None
        self.input_spec = layers.InputSpec(min_ndim=4,
                                           axes={-1: input_channel})
        self.built = True

    def call(self, inputs, *args, **kwargs):
        inputs, padding = pad_for_chip(inputs, self.strides, self.kernel_size)
        strides = (1,) + self.strides + (1,)

        if self.quantize:
            outputs = quantized_conv2d(self.specification,
                                       inputs=inputs,
                                       kernel=self.kernel,
                                       biases=self.bias,
                                       bit_mask=self.bit_mask,
                                       strides=self.strides,
                                       padding=padding)
        else:
            outputs = tf.nn.conv2d(inputs, self.kernel, strides, padding)
            if self.use_bias:
                outputs = tf.nn.bias_add(outputs, self.bias)

        return outputs

    def get_config(self):
        config = super(Conv2D, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'use_bias': self.use_bias,
            'quantize': self.quantize,
            'bit_mask': self.bit_mask,
            'chip_id': self.chip_id,
            'clip_bias': self.clip_bias
        })
        return config


class Conv2DTranspose(Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 chip_id,
                 upsample_filling_mode,
                 strides=(1, 1),
                 use_bias=True,
                 quantize=False,
                 bit_mask=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 clip_bias=False,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(Conv2DTranspose, self).__init__(filters,
                                              kernel_size,
                                              chip_id,
                                              strides,
                                              use_bias,
                                              quantize,
                                              bit_mask,
                                              kernel_initializer,
                                              bias_initializer,
                                              clip_bias,
                                              trainable,
                                              name,
                                              **kwargs)

        self.upsample_filling_mode = upsample_filling_mode
        self.upsample = layers.UpSampling2D(size=(2, 2), data_format='channels_last')

    def call(self, inputs, *args, **kwargs):
        outputs = self.upsample(inputs)
        if self.upsample_filling_mode == UpSampleFillingMode.ZERO:
            overlay = tf.constant([[1, 0], [0, 0]], dtype=tf.float32)
            overlay = tf.tile(overlay, (tf.shape(outputs)[1] / 2, tf.shape(outputs)[2] / 2))
            overlay = tf.reshape(overlay, shape=(1, tf.shape(overlay)[0], tf.shape(overlay)[1], 1))
            outputs *= overlay
        return super(Conv2DTranspose, self).call(outputs, *args, **kwargs)

    def get_config(self):
        config = super(Conv2DTranspose, self).get_config()
        config.update({
            'upsample_filling_mode': self.upsample_filling_mode
        })
        return config
