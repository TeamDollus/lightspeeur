import tensorflow as tf

from tensorflow.keras import layers
from lightspeeur.drivers import constant_initializer


class BatchNormalization(layers.Layer):
    def __init__(self,
                 trainable=True,
                 beta_initializer=None,
                 gamma_initializer=None,
                 moving_mean_initializer=None,
                 moving_variance_initializer=None,
                 name=None,
                 **kwargs):

        super(BatchNormalization, self).__init__(trainable=trainable, name=name, **kwargs)

        self.beta_initializer = constant_initializer(beta_initializer, tf.initializers.zeros())
        self.gamma_initializer = constant_initializer(gamma_initializer, tf.initializers.ones())
        self.moving_mean_initializer = constant_initializer(moving_mean_initializer, tf.initializers.zeros())
        self.moving_variance_initializer = constant_initializer(moving_variance_initializer, tf.initializers.ones())

    def call(self, inputs, *args, **kwargs):
        return layers.BatchNormalization(beta_initializer=self.beta_initializer,
                                         gamma_initializer=self.gamma_initializer,
                                         moving_mean_initializer=self.moving_mean_initializer,
                                         moving_variance_initializer=self.moving_variance_initializer)(inputs)
