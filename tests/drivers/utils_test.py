import pytest
import numpy as np
import tensorflow as tf

from lightspeeur.drivers import permute_axis, conversion_initializer,  constant_initializer


def test_permute_axis():
    a = np.random.rand(3, 4, 5)
    t_einsum = np.einsum('ijk->kji', a)
    t_driver = a.transpose(permute_axis('IJK', 'KJI'))
    assert a.shape != t_einsum.shape
    assert a.shape != t_driver.shape
    assert t_einsum.shape == t_driver.shape
    assert np.allclose(t_einsum, t_driver)


def test_conversion_initializer():
    M = np.random.rand(3, 4, 5)
    initializer = tf.initializers.zeros()
    default_initializer = tf.initializers.glorot_uniform()
    res = conversion_initializer(None, default_initializer)
    assert res == default_initializer
    res = conversion_initializer(initializer, default_initializer)
    assert res == initializer
    res = conversion_initializer(M, default_initializer)
    assert np.allclose(res.numpy(), M)


def test_constant_initializer():
    M = np.random.rand(3, 4, 5)
    initializer = tf.initializers.zeros()
    default_initializer = tf.initializers.glorot_uniform()
    res = constant_initializer(None, default_initializer)
    assert res == default_initializer
    res = constant_initializer(initializer, default_initializer)
    assert res == initializer
    res = constant_initializer(M, default_initializer)
    assert np.allclose(res(shape=M.shape), M)


if __name__ == '__main__':
    pytest.main([__file__])
