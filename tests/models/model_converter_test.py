import os
import json
import pytest

from lightspeeur.models import ModelConverter
from lightspeeur.models.model_converter import is_convolutional, is_pooling
from lightspeeur.drivers import UpSampleFillingMode
from lightspeeur.layers import Conv2D, Conv2DTranspose, DepthwiseConv2D, ReLU, MaxPooling2D, TopLeftPooling2D
from tensorflow.keras import Model, layers


def build_simple_conv_model():
    inputs = layers.Input(shape=(28, 28, 1))
    x = inputs
    # on-chip layers
    filters = [32, 64, 128]
    graph = []
    for i in range(3):
        chunk = ['conv{}_conv', 'conv{}_relu', 'conv{}_pooling']
        chunk = [name.format(i) for name in chunk]

        x = Conv2D(filters[i], (3, 3), '2803', (1, 1), use_bias=False, bit_mask=12, name=chunk[0])(x)
        x = ReLU('2803', name=chunk[1])(x)
        x = MaxPooling2D(name=chunk[2])(x)

        graph.append(chunk)

    # off-chip layers
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x, name='simple_conv_model')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model, graph


def build_simple_upsample_conv_model():
    inputs = layers.Input(shape=(224, 224, 1))
    x = inputs

    # on-chip layers
    blocks = [(16, 2, False, True),  # filters, blocks, upsample, pool
              (32, 2, False, True),
              (32, 2, False, True),
              (64, 2, False, False),
              (32, 2, True, False),
              (16, 2, True, False),
              (4, 2, False, False)]

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

    model = Model(inputs=inputs, outputs=x, name='simple_upsample_conv')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model, graph


def check_model_sameness(model, graph, body):
    for layer_info in body['layer']:
        chunk = graph[layer_info['major_layer'] - 1]

        coef_bits = []
        num_sublayers = 0
        pooling = False
        depthwise_conv = False
        transpose_conv = False
        image_size = None
        input_channels, output_channels = None, None

        for i, layer_name in enumerate(chunk):
            layer = model.get_layer(name=layer_name)
            if is_convolutional(layer):
                if input_channels is None:
                    input_channels = layer.input.shape[-1]
                output_channels = layer.output.shape[-1]

                if isinstance(layer, Conv2DTranspose):
                    layer_image_size = layer.output.shape[1]
                else:
                    layer_image_size = layer.input.shape[1]
                if image_size is None:
                    image_size = layer_image_size
                else:
                    assert image_size == layer_image_size

                coef_bits.append(layer.bit_mask)
                num_sublayers += 1

                if isinstance(layer, DepthwiseConv2D):
                    depthwise_conv = True
                elif isinstance(layer, Conv2DTranspose):
                    transpose_conv = True

            elif is_pooling(layer):
                pooling = True
                assert i == len(chunk) - 1  # the pooling layer must be at the end of the major layer

        assert layer_info['image_size'] == image_size
        assert layer_info['input_channels'] == input_channels
        assert layer_info['output_channels'] == output_channels

        assert len(set(coef_bits)) == 1  # only one bit mask can be existed
        assert layer_info['coef_bits'] == set(coef_bits).pop()  # the bit mask must equals with coef_bits
        assert layer_info['sublayer_number'] == num_sublayers  # number of sublayers are must equal
        if layer_info.get('pooling'):
            assert layer_info['pooling'] == pooling
        else:
            assert not pooling

        if layer_info.get('depth_enable'):
            assert layer_info['depth_enable'] == depthwise_conv
        else:
            assert not depthwise_conv

        if layer_info.get('upsample_enable'):
            assert layer_info['upsample_enable'] == transpose_conv
        else:
            assert not transpose_conv
    return True


def check_model_data_generation(model, graph):
    converter = ModelConverter('2803', model, graph={model.name: graph}, debug=True)

    tmp_file = 'model_data.json'
    converter.create_default_model_data(converter.spec.find_proper_device(), tmp_file, graph)

    with open(tmp_file, 'r') as f:
        body = json.load(f)

    assert check_model_sameness(model, graph, body)
    os.remove(tmp_file)
    return True


def test_model_data_generation():
    assert check_model_data_generation(*build_simple_conv_model())
    assert check_model_data_generation(*build_simple_upsample_conv_model())


if __name__ == '__main__':
    pytest.main([__file__])
