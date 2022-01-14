import os
import json
import logging
import shutil

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


class ModelConverter:

    def __init__(self, chip_id: str, model: Model, graph: dict, config_path='bin/libgticonfig2803.so', debug=False):
        self.chip_id = chip_id
        self.spec = Specification(chip_id=chip_id)
        self.model = model
        self.graph = graph
        self.config_path = config_path
        self.debug = debug
        self.working_files = []

    def convert(self, driver: Driver):
        self.working_files = []
        results = []
        for graph_name, small_graph in self.graph.items():
            logger.info('Preparing conversion of {}'.format(graph_name))
            data_infos = self.convert_on_chip_graph(graph_name, small_graph)
            # TODO: Really host layer required?
            for i, data_info in enumerate(data_infos):
                model_def = self.update_model_definition(graph_name=graph_name,
                                                         model_file='{}_model.json'.format(graph_name),
                                                         dst_model_file='{}_dst_model.json'.format(graph_name),
                                                         data_info=data_info)
                if len(data_infos) == 1:
                    outputs = '{}.model'.format(graph_name)
                else:
                    outputs = '{}_{}.model'.format(graph_name, i)

                with driver:
                    driver.compose(model_def, outputs)

                self.working_files.append(outputs)
                data_info['dst_model_def'] = os.path.relpath(model_def)
                if len(data_infos) == 1:
                    logger.info('Successfully generated \'{}\' for \'{}\''.format(outputs, graph_name))
                    target_dir = '{}_target'.format(graph_name)
                else:
                    logger.info('Successfully generated {}th \'{}\' for \'{}\''.format(i + 1, outputs, graph_name))
                    target_dir = '{}_{}_target'.format(graph_name, i)
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)

                os.makedirs(target_dir, exist_ok=True)
                for tracked_file in self.working_files:
                    shutil.move(tracked_file, os.path.join(target_dir, tracked_file))
                logger.info('All working files have been moved to \'{}\''.format(target_dir))
                results.append(os.path.join(target_dir, outputs))
        return results

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

            bit_shift = compute_quantized_shift(self.spec, kernel, bias, layer.bit_mask)
            kernel = quantize_kernel(kernel, bit_shift, layer.bit_mask)
            kernel = kernel.numpy().transpose(tuple(permute_axis('HWIO', 'OIHW')))
            bias = quantized_shift(bias, bit_shift)

            if self.debug:
                logger.info('Layer: {}'.format(layer_name))
                logger.info('max(abs(W)): {}, max(abs(B)): {}, shift: {}'
                            .format(np.amax(np.abs(kernel)),
                                    np.amax(np.abs(bias)),
                                    bit_shift))
                logger.info('')

            kernels = np.concatenate((kernels, np.ravel(kernel)))
            biases = np.concatenate((biases, np.ravel(bias)))
            bit_shifts.append(int(bit_shift))

        logger.info('Converting convolutional layers to data file')

        kernels_filepath = '{}_kernels'.format(graph_name)
        biases_filepath = '{}_biases'.format(graph_name)
        data_filepath = '{}_data.json'.format(graph_name)
        dst_data_filepath = '{}_dst_data.json'.format(graph_name)

        kernels.tofile(kernels_filepath, sep='\n', format='%.16e')
        biases.tofile(biases_filepath, sep='\n', format='%.16e')

        self.working_files.append(kernels_filepath)
        self.working_files.append(biases_filepath)

        dst_data_file = self.update_model_data(data_filepath, dst_data_filepath, bit_shifts, small_graph=small_graph)
        dst_chip_file = '{}_chip'.format(graph_name)

        self.working_files.append(dst_chip_file)

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

                    self.working_files.append(configurer.get_work_dir())
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
        self.working_files.append(dst_data_file)

        return dst_data_file

    def update_model_definition(self, graph_name, model_file, dst_model_file, data_info):
        if not os.path.exists(model_file):
            self.create_default_model_definition(graph_name, model_file)

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
                layer['data file'] = data_info['dat{}'.format(data_index)]
                cascade = data_index < num_conv - 1
                self.configure_conv_model(conv=layer,
                                          data_file=data_info['datjson{}'.format(data_index)],
                                          cascade=cascade)
                data_index += 1
            elif op == 'LABEL':
                layer['data file'] = data_info['label']
            elif op == 'FC':
                layer['data file'] = data_info[layer['name']]

        with open(dst_model_file, 'w') as f:
            json.dump(body, f, indent=4, sort_keys=True)
        self.working_files.append(dst_model_file)

        return dst_model_file

    def configure_conv_model(self, conv, data_file, cascade):
        if cascade:
            raise NotImplementedError('Cascade mode on this converter is not supported yet.')

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

    def create_default_model_data(self, data_file, small_graph):
        layers = []
        for major_index, chunk in enumerate(small_graph):
            layer_info = {
                'major_layer': major_index + 1
            }

            input_channels = None
            output_channels = None
            image_size = None
            coef_bits = None

            conv_included = False
            one_coefficients = []
            excluded_layers = 0
            for layer_name in chunk:
                layer = self.model.get_layer(layer_name)
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
                    elif image_size != layer_image_size:
                        raise ValueError('All graph chunk must have same image size. {} != {}'
                                         .format(image_size, layer_image_size))

                    if coef_bits is None:
                        coef_bits = layer.bit_mask
                    elif coef_bits != layer.bit_mask:
                        raise ValueError('All graph chunk must have same coefficients bit size: {} != {}'
                                         .format(coef_bits, layer.bit_mask))

                    if isinstance(layer, DepthwiseConv2D):
                        layer_info['depth_enable'] = True
                    if isinstance(layer, Conv2DTranspose):
                        layer_info['upsample_enable'] = True
                    one_coefficients.append(1 if layer.kernel_size == (1, 1) else 0)
                    conv_included = True
                elif is_pooling(layer):
                    layer_info['pooling'] = True
                    excluded_layers += 1
                else:
                    excluded_layers += 1
            if not conv_included:
                raise ValueError('Graph layer chunk must include at least one or more convolutional layers')
            if not input_channels:
                raise ValueError('Input channel information is not valid')
            if not output_channels:
                raise ValueError('Output channel information is not valid')
            if not image_size:
                raise ValueError('Image size information is not valid')

            num_sublayers = len(chunk) - excluded_layers
            layer_info['sublayer_number'] = num_sublayers
            if num_sublayers > 0:
                layer_info['one_coef'] = one_coefficients
            layer_info['input_channels'] = input_channels
            layer_info['output_channels'] = output_channels
            layer_info['image_size'] = image_size
            layer_info['coef_bits'] = coef_bits
            layers.append(layer_info)

        body = {
            'layer': layers,
            'model': [{
                'ChipType': int(self.chip_id),
            }]
        }
        with open(data_file, 'w') as f:
            json.dump(body, f, indent=4, sort_keys=True)
        self.working_files.append(data_file)
        logger.info('Generated default model data file')

    def create_default_model_definition(self, graph_name, model_file):
        latest_chunk = self.graph[graph_name][-1]
        output_shapes = (-1, 1, 1)  # channels, height, width
        for layer_name in latest_chunk:
            layer = self.model.get_layer(layer_name)
            if is_convolutional(layer):
                shape = layer.output.shape  # batch_size, width, height, channels
                output_shapes = (shape[-1], shape[-2], shape[-3])
                break

        if output_shapes[0] == -1:
            raise ValueError('The latest graph layer chunk does not contain proper output shapes')

        # multiple layers for multiple-chip mode (cascade mode)
        body = {
            'name': graph_name,
            'layer': [{
                'data offset': 0,
                'device': {
                    'chip': self.chip_id,
                    'name': None,
                    'type': 0
                },
                'mode': 0,
                'name': 'cnn',
                'operation': 'GTICNN',
                'output channels': output_shapes[0],
                'output height': output_shapes[1],
                'output width': output_shapes[2]
            }]
        }
        with open(model_file, 'w') as f:
            json.dump(body, f, indent=4, sort_keys=True)
        self.working_files.append(model_file)
        logger.info('Generated default model definition file')
