import tensorflow as tf


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
        return tf.constant_initializer(initializer)
