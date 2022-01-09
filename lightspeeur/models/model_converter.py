import os
import json
import logging

import numpy as np

from tensorflow.keras import Model
from lightspeeur.drivers.specification import Specification
from lightspeeur.drivers import Configurer, Driver
from lightspeeur.layers import Conv2D, Conv2DTranspose, DepthwiseConv2D, MaxPooling2D, TopLeftPooling2D
from lightspeeur.layers.quantization import quantize_kernel, quantized_shift, compute_quantized_shift
from lightspeeur.drivers.utils import permute_axis

logger = logging.getLogger('lightspeeur')
logger.setLevel(logging.INFO)
logging.basicConfig()


example_graphs = {
    'convs1': [['conv1/conv', 'conv1/batch_norm', 'conv2/relu'],
               ['conv2/conv', 'conv2/relu']]
}

ELIGIBLE_CONVS = [Conv2D, Conv2DTranspose, DepthwiseConv2D]
ELIGIBLE_POOLINGS = [MaxPooling2D, TopLeftPooling2D]


def is_convolutional(layer):
    return any(isinstance(layer, conv) for conv in ELIGIBLE_CONVS)


def is_pooling(layer):
    return any(isinstance(layer, pooling) for pooling in ELIGIBLE_POOLINGS)


def require_residual_layers(layer):
    return (layer.get('resnet_shortcut_start_layers')
            and layer.get('pooling')
            and layer['sublayer_number'] == layer['resnet_shortcut_start_layers'][-1] + 1)


def configure_conv_model(conv, data_file, cascade):
    with open(data_file, 'r') as f:
        body = json.load(f)

    models = body['model']
    for model in models:
        chip_id = model['ChipType']
        spec = Specification(chip_id)
        min_channels = spec.min_channels()

        layers = body['layer']
        for i, layer in enumerate(layers):
            if i == 0:
                conv['inputs'] = [{
                    'format': 'byte',
                    'prefilter': 'interlace_tile_encode',
                    'shape': [layer['image_size'], layer['image_size'], layer['input_channels']]
                }]
                conv['outputs'] = []

            format_type = 'byte'
            postfilter = 'interlace_tile_decode'
            image_size = layer['image_size']
            out_channels = max(layer['output_channels'], min_channels)
            upsample = False

            if layer.get('upsample_enable'):
                image_size *= 2
                upsample = True

            if layer.get('ten_bits_enable'):
                format_type = 'float'
                postfilter = 'interlace_tile_10bits_decode'

            outputs_size = image_size * image_size * out_channels * 32 // 49
            if layer.get('learning'):
                num_sublayers = layer['sublayer_number']
                if require_residual_layers(layer):
                    num_sublayers += 1

                for _ in range(num_sublayers):
                    outputs = {
                        'format': format_type,
                        'postfilter': postfilter,
                        'shape': [image_size, image_size, out_channels, min(image_size, 14) ** 2, outputs_size]
                    }
                    if upsample:
                        outputs['upsampling'] = 1

                    conv['outputs'].append(outputs)
            elif layer.get('last_layer_out'):
                outputs = {
                    'format': format_type,
                    'postfilter': postfilter,
                    'shape': [image_size, image_size, out_channels, min(image_size, 14) ** 2, outputs_size]
                }
                if upsample:
                    outputs['upsampling'] = 1

                conv['outputs'].append(outputs)
            elif i == len(layers) - 1:
                byte_scale = 32
                if layer.get('pooling'):
                    image_size //= 2
                    if image_size == 7:
                        postfilter = 'fc77_decode'
                        byte_scale = 64

                outputs = {
                    'format': format_type,
                    'postfilter': postfilter,
                    'shape': [image_size, image_size,
                              out_channels,
                              min(image_size, 14) ** 2,
                              image_size * image_size * out_channels * byte_scale // 49]
                }
                if postfilter == 'fc77_decode':
                    outputs['layer scaledown'] = 3
                elif postfilter == 'interlace_tile_10bits_decode':
                    outputs['shape'][-1] *= 2

                if upsample:
                    outputs['upsampling'] = 1

                if cascade:
                    if 'layer scaledown' not in outputs:
                        outputs['layer scaledown'] = -3
                    else:
                        outputs['layer scaledown'] -= 3

                conv['outputs'].append(outputs)


def update_model_definition(model_file, dst_model_file, data_info):
    with open(model_file, 'r') as f:
        body = json.load(f)

    num_conv = 0
    for layer in body['layer']:
        if layer['operation'] == 'GTICNN':
            num_conv += 1

    # SDK > 4 requires version 100
    if 'version' not in body:
        body['version'] = 100

    data_index = 0
    for layer in body['layer']:
        op = layer['operation']
        if op == 'GTICNN':
            layer['data file'] = data_info['data{}'.format(data_index)]
            cascade = data_index < num_conv - 1
            configure_conv_model(conv=layer,
                                 data_file=data_info['datjson{}'.format(data_index)],
                                 cascade=cascade)
            data_index += 1
        elif op == 'LABEL':
            layer['data file'] = data_info['label']
        elif op == 'FC':
            layer['data file'] = data_info[layer['name']]

    with open(dst_model_file, 'w') as f:
        json.dump(body, f, indent=4, sort_keys=True)

    return dst_model_file


class ModelConverter:

    def __init__(self, chip_id: str, model: Model, graph: dict, config_path: str, debug=False):
        self.chip_id = chip_id
        self.spec = Specification(chip_id=chip_id)
        self.model = model
        self.graph = graph
        self.config_path = config_path
        self.debug = debug

    def convert(self, driver: Driver):
        for graph_name, small_graph in self.graph.items():
            print('Preparing conversion of {}'.format(graph_name))
            data_infos = self.convert_on_chip_graph(graph_name, small_graph)
            # TODO: Really host layer required?
            for i, data_info in enumerate(data_infos):
                model_def = update_model_definition(model_file='{}_model.json'.format(graph_name),
                                                    dst_model_file='{}_dst_model.json'.format(graph_name),
                                                    data_info=data_info)
                if len(data_infos) == 1:
                    outputs = '{}.model'.format(graph_name)
                else:
                    outputs = '{}_{}.model'.format(graph_name, i)

                with driver:
                    driver.compose(model_def, outputs)

                data_info['dst_model_def'] = os.path.relpath(model_def)
                if len(data_infos) == 1:
                    print('Successfully generated \'{}\' for \'{}\''.format(outputs, graph_name))
                else:
                    print('Successfully generated {}th \'{}\' for \'{}\''.format(i + 1, outputs, graph_name))

    def convert_on_chip_graph(self, graph_name, small_graph):
        kernels = np.array([])
        biases = np.array([])
        bit_shifts = []

        flattened = [item for chunk in small_graph for item in chunk]
        for layer_name in flattened:
            layer = self.model.get_layer(layer_name)
            if not is_convolutional(layer):
                continue

            weights = layer.get_weights()
            if len(weights) != 2:
                raise ValueError('Each convolutional layers must have kernel and bias.')

            kernel = weights[0]
            bias = weights[1]

            bit_shift = compute_quantized_shift(self.spec, kernel, bias, layer.mask_bit)
            kernel = quantize_kernel(kernel, bit_shift, layer.mask_bit)
            kernel = kernel.transpose(tuple(permute_axis('HWIO', 'OIHW')))
            bias = quantized_shift(bias, bit_shift)

            if self.debug:
                logger.info('Layer: {}, {}'.format(layer_name, weights[0]))
                logger.info('max(abs(W)): {}, max(abs(B)): {}, shift: {}'
                            .format(np.amax(np.abs(kernel)),
                                    np.amax(np.abs(bias)),
                                    bit_shift))
                logger.info('')

            kernels = np.concatenate((kernels, kernel.ravel()))
            biases = np.concatenate((biases, bias.ravel()))
            bit_shifts.append(int(bit_shift))

        logger.info('Converting convolutional layers to data file')

        kernels_filepath = '{}_kernels'.format(graph_name)
        biases_filepath = '{}_biases'.format(graph_name)
        data_filepath = '{}_data.json'.format(graph_name)
        dst_data_filepath = '{}_dst_data.json'.format(graph_name)

        kernels.tofile(kernels_filepath, sep='\n', format='%.16e')
        biases.tofile(biases_filepath, sep='\n', format='%.16e')
        dst_data_file = self.update_model_data(data_filepath, dst_data_filepath, bit_shifts, small_graph=small_graph)
        dst_chip_file = '{}_chip'.format(graph_name)

        data_info = []
        with open(dst_data_file, 'r') as f:
            body = json.load(f)
            models = body['model']
            for model in models:
                if 'ChipNumber' in model and model['ChipNumber'] > 1:
                    raise ValueError('Multi-chip model is currently not supported')
                else:
                    configurer = Configurer(graph_name, self.config_path,
                                            kernels_filepath, biases_filepath,
                                            dst_data_file, dst_chip_file,
                                            debug=self.debug)
                    with configurer:
                        configurer.configure()

                    data_info.append({
                        'dat0': dst_chip_file,
                        'datjson0': dst_data_file,
                        'filter': kernels_filepath,
                        'bias': biases_filepath
                    })
        return data_info

    def update_model_data(self, data_file, dst_data_file, new_shifts, small_graph=None):
        if not os.path.exists(data_file):
            if small_graph is None:
                raise AttributeError('Graph is required when data file is not existed')
            self.create_default_model_data(data_file, small_graph)
            pass

        with open(data_file, 'r') as f:
            body = json.load(f)
            for models in body['model']:
                models['MajorLayerNumber'] = len(body['layer'])

            offset = 0
            for major_layer in body['layer']:
                sublayers = major_layer['sublayer_number']
                major_layer['scaling'] = new_shifts[offset:offset + sublayers]
                offset += sublayers

        with open(dst_data_file, 'w') as f:
            json.dump(body, f, indent=4, sort_keys=True)

        return dst_data_file

    def create_default_model_data(self, data_file, small_graph):
        layers = []
        for major_index, chunk in enumerate(small_graph):
            layer_info = {}
            for layer_name in chunk:
                layer = self.model.get_layer(layer_name)
                if is_convolutional(layer):
                    layer_info['coef_bits'] = layer.bit_mask
                    layer_info['depth_enable'] = isinstance(layer, DepthwiseConv2D)
                    layer_info['major_layer'] = major_index + 1
                    layer_info['image_size'] = layer.input.shape[1]
                    layer_info['input_channels'] = layer.input.shape[-1]
                    layer_info['output_channels'] = layer.output.shape[-1]
                elif is_pooling(layer):
                    layer_info['pooling'] = True
            layer_info['sublayer_number'] = len(chunk) - 1  # exclude one conv layer
            layer_info['one_coef'] = []
            layers.append(layer_info)

        body = {
            'layer': layers,
            'model': [{
                'ChipType': self.chip_id,
            }]
        }
        with open(data_file, 'w') as f:
            json.dump(body, f, indent=4, sort_keys=True)
