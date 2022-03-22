import os
import json
import pytest

from lightspeeur.models import ModelConverter
from lightspeeur.drivers import UpSampleFillingMode
from lightspeeur.layers import Conv2D, Conv2DTranspose, ReLU, MaxPooling2D, TopLeftPooling2D
from tensorflow.keras import Model, layers


def model_unlike_image_size():
    # there are 2 max pooling layer in a single major layer
    inputs = layers.Input(shape=(28, 28, 1))
    x = inputs
    x = Conv2D(16, (3, 3), '2803', use_bias=False, bit_mask=12, name='block1/conv')(x)
    x = ReLU('2803', name='block1/relu')(x)
    x = MaxPooling2D(name='block1/pool')(x)
    x = Conv2D(32, (3, 3), '2803', use_bias=False, bit_mask=12, name='block2/conv')(x)
    x = ReLU('2803', name='block2/relu')(x)
    x = MaxPooling2D(name='block2/pool')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=x, name='unlike_image_size')
    return model, [['block1/conv', 'block1/relu', 'block1/pool',
                    'block2/conv', 'block2/relu', 'block2/pool']]


def model_unlike_coefficient_bit_size():
    # `block1/conv1` have 12 of bit mask size, `block1/conv2` have 8 of bit mask size
    inputs = layers.Input(shape=(28, 28, 1))
    x = inputs
    x = Conv2D(16, (3, 3), '2803', use_bias=False, bit_mask=12, name='block1/conv1')(x)
    x = ReLU('2803', name='block1/relu1')(x)
    x = Conv2D(16, (3, 3), '2803', use_bias=False, bit_mask=8, name='block1/conv2')(x)
    x = ReLU('2803', name='block1/relu2')(x)
    x = MaxPooling2D(name='block1/pool')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=x, name='unlike_coefficient_bit_size')
    return model, [['block1/conv1', 'block1/relu1', 'block1/conv2', 'block2/relu2', 'block1/pool']]


def model_convolutional_layer_not_included():
    # no convolutional layer in small graph chunk
    inputs = layers.Input(shape=(28, 28, 1))
    x = inputs
    x = MaxPooling2D(name='block/pool')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=x, name='convolutional_layer_not_included')
    return model, [['block/pool']]


def model_2803_invalid_model_input_size():
    # 128x128 -> 64x64
    inputs = layers.Input(shape=(128, 128, 1))
    x = inputs
    x = Conv2D(16, (3, 3), '2803', use_bias=False, bit_mask=12, name='block/conv')(x)
    x = ReLU('2803', name='block/relu')(x)
    x = MaxPooling2D(name='block/pool')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=x, name='2803_invalid_model_input_size')
    return model, [['block/conv', 'block/relu', 'block/pool']]


def model_2803_invalid_model_output_size():
    # 448x448 -> 896x896
    inputs = layers.Input(shape=(448, 448, 1))
    x = inputs
    x = Conv2DTranspose(16, (3, 3), '2803',
                        upsample_filling_mode=UpSampleFillingMode.ZERO,
                        use_bias=False, bit_mask=12, name='block/upsample')(x)
    x = ReLU('2803', name='block/relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=x, name='2803_invalid_model_output_size')
    return model, [['block/upsample', 'block/relu']]


def model_2803_too_many_major_layers():
    # 2803 supports up to 6 major layers, but the model have 7 major layers
    inputs = layers.Input(shape=(224, 224, 1))
    x = inputs

    # on-chip layers
    blocks = [(16, 2, False, True),  # filters, blocks, upsample, pool
              (32, 2, False, True),
              (32, 2, False, True),
              (64, 2, False, False),
              (32, 2, True, False),
              (16, 2, True, False),
              (16, 2, False, False)]

    def _conv_block(block_inputs, filters, upsample=False, name='conv2d'):
        names = []
        outputs = block_inputs
        if upsample:
            outputs = Conv2DTranspose(filters, (3, 3),
                                      '2803',
                                      UpSampleFillingMode.ZERO,
                                      (1, 1),
                                      use_bias=False, bit_mask=12,
                                      name='{}_deconv'.format(name))(outputs)
            names.append('{}_deconv'.format(name))
        else:
            outputs = Conv2D(filters, (3, 3),
                             '2803',
                             (1, 1),
                             use_bias=False, bit_mask=12,
                             name='{}_conv'.format(name))(outputs)
            names.append('{}_conv'.format(name))
        outputs = ReLU('2803', name='{}_relu'.format(name))(outputs)
        names.append('{}_relu'.format(name))
        return outputs, names

    graph = []
    for index, block in enumerate(blocks):
        chunk = []
        for block_index in range(block[1]):
            x, names = _conv_block(x, block[0], block_index == 0 and block[2], name='conv2d_{}_{}'.format(index, block_index))
            chunk += names
        if block[3]:
            x = TopLeftPooling2D(name='top_left_pooling_{}'.format(index))(x)
            chunk.append('top_left_pooling_{}'.format(index))
        graph.append(chunk)

    x = layers.Flatten()(x)
    x = layers.Dense(4096, name='activation')(x)
    x = layers.Dense(10, name='softmax')(x)

    model = Model(inputs=inputs, outputs=x, name='2803_too_many_major_layers')
    return model, graph


def model_2803_too_many_sub_layers():
    # 2803 supports up to 12 sublayers in a single major layer
    inputs = layers.Input(shape=(28, 28, 1))
    x = inputs
    chunk = []
    for i in range(13):
        conv_name = 'block/conv{}'.format(i)
        relu_name = 'block/relu{}'.format(i)
        chunk.append(conv_name)
        x = Conv2D(64, (3, 3), '2803',
                   use_bias=False, bit_mask=12, name=conv_name)(x)
        x = ReLU('2803', name=relu_name)(x)
    pool_name = 'block/pool'
    chunk.append(pool_name)

    x = MaxPooling2D(name=pool_name)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(10, name='softmax')(x)

    model = Model(inputs=inputs, outputs=x, name='2803_too_many_sub_layers')
    return model, [chunk]


def test_model_conversion():
    assert evaluate_model_availability(model_unlike_image_size(), ValueError)
    assert evaluate_model_availability(model_unlike_coefficient_bit_size(), ValueError)
    assert evaluate_model_availability(model_convolutional_layer_not_included(), ValueError)
    assert evaluate_model_availability(model_2803_invalid_model_input_size(), ValueError)
    assert evaluate_model_availability(model_2803_invalid_model_output_size(), ValueError)
    assert evaluate_model_availability(model_2803_too_many_major_layers(), ValueError)
    assert evaluate_model_availability(model_2803_too_many_sub_layers(), ValueError)


def evaluate_model_availability(model_graph, error_type=None):
    model, graph = model_graph
    converter = ModelConverter('2803', model, graph={model.name: graph}, debug=True)
    filename = 'model_{}.json'.format(model.name)

    def gen():
        device = converter.spec.find_proper_device()
        converter.check_model_graph_spec(graph, device)
        converter.create_default_model_data(device, filename, graph)

    if error_type is not None:
        with pytest.raises(error_type):
            gen()
    else:
        gen()

    if os.path.exists(filename):
        os.remove(filename)
    return True


if __name__ == '__main__':
    pytest.main([__file__])
