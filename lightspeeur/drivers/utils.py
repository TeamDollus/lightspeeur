import tensorflow as tf

from lightspeeur.drivers.specification import Specification


def permute_axis(axis_from, axis_to):
    return list(map(lambda x: axis_from.find(x), axis_to))


def conversion_initializer(initializer, default_initializer):
    if initializer is None:
        return default_initializer
    elif callable(initializer):
        return initializer
    else:
        return tf.convert_to_tensor(initializer, tf.float32)


def constant_initializer(initializer, default_initializer):
    if initializer is None:
        return default_initializer
    elif callable(initializer):
        return initializer
    else:
        return tf.constant_initializer(initializer, verify_shape=True)


def validate_image_fit_to_chip(buffer, size_axis=0):
    shape = buffer[size_axis]
    if shape not in Specification.ALLOWED_IMAGE_SIZES:
        raise ValueError('Incompatible input image size: {}x{}'.format(shape, shape))
    return buffer
