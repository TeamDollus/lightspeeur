import pytest

from tensorflow.keras import models, layers
from lightspeeur.layers import Conv2D, ReLU
from lightspeeur.models import ModelStageAdvisor


def mock_convolutional_model(use_bias=False, conv_q=False, activation_q=False, batch_norm=False):
    inputs = layers.Input(shape=(224, 224, 3))
    x = inputs
    x = Conv2D(32, kernel_size=(3, 3), chip_id='2803', use_bias=use_bias, quantize=conv_q, bit_mask=5)(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = ReLU('2803', quantize=activation_q)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inputs=inputs, outputs=x, name='convolutional_model_mock')


def test_analog_conv_has_bias():
    model = mock_convolutional_model(use_bias=True, conv_q=False)
    with pytest.raises(ValueError):
        ModelStageAdvisor('2803', model, {})


def test_quantized_conv_has_no_bias():
    model = mock_convolutional_model()
    folded_model = mock_convolutional_model(use_bias=False, conv_q=True)
    with pytest.raises(ValueError):
        ModelStageAdvisor('2803', model, {}, folded_shape_model=folded_model)


def test_analog_conv_however_folded():
    model = mock_convolutional_model()
    folded_model = mock_convolutional_model(use_bias=True, conv_q=False)
    with pytest.raises(ValueError):
        ModelStageAdvisor('2803', model, {}, folded_shape_model=folded_model)


def test_analog_activation_however_folded():
    model = mock_convolutional_model()
    folded_model = mock_convolutional_model(use_bias=True, conv_q=True, activation_q=False)
    with pytest.raises(ValueError):
        ModelStageAdvisor('2803', model, {}, folded_shape_model=folded_model)


def test_folded_model_has_batch_norm():
    model = mock_convolutional_model()
    folded_model = mock_convolutional_model(use_bias=True, conv_q=True, activation_q=True, batch_norm=True)
    with pytest.raises(ValueError):
        ModelStageAdvisor('2803', model, {}, folded_shape_model=folded_model)


if __name__ == '__main__':
    pytest.main([__file__])