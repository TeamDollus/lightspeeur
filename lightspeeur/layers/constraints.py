import tensorflow as tf

from tensorflow.keras.constraints import Constraint


class ClippingBiasConstraint(Constraint):

    def __init__(self, kernel):
        self.max_magnitude = tf.reduce_max(tf.abs(kernel))

    def __call__(self, weights):
        return tf.clip_by_value(weights, -self.max_magnitude, self.max_magnitude)
