from enum import Enum

import os
import json


class UpSampleFillingMode(Enum):
    REPEAT = 1
    ZERO = 2


SPEC_INFO = None


class DeviceInfo:

    def __init__(self,
                 chip_id,
                 allowed_input_size, allowed_output_size, allowed_channels,
                 major_layer_limit, sub_layer_limit):

        self.chip_id = chip_id

        self.allowed_input_size = allowed_input_size
        self.allowed_output_size = allowed_output_size
        self.allowed_channels = allowed_channels

        self.major_layer_limit = major_layer_limit
        self.sub_layer_limit = sub_layer_limit
        self.total_layer_limit = major_layer_limit * sub_layer_limit


class Specification:

    REGISTERED_DEVICES = [
        DeviceInfo('2803',
                   allowed_input_size=[448, 224, 112, 56, 28, 14],
                   allowed_output_size=[448, 224, 112, 56, 28, 14],
                   allowed_channels=[16, 32, 64, 128, 256, 512, 1024],
                   major_layer_limit=6,
                   sub_layer_limit=12),
        DeviceInfo('5801',
                   allowed_input_size=[896, 448, 224, 112, 56, 28, 14],
                   allowed_output_size=[896, 448, 224, 112, 56, 28, 14, 7],
                   allowed_channels=[4, 8, 16, 32, 64, 128, 256, 512, 1024],
                   major_layer_limit=7,
                   sub_layer_limit=12)
    ]

    DEFAULT_IMAGE_SIZE = 224
    ALLOWED_IMAGE_SIZES = [224, 448, 320, 640]

    DEFAULT_CONV_KERNEL_SIZE = (3, 3)
    ALLOWED_CONV_KERNEL_SIZES = [(1, 1), (3, 3),
                                 [1, 1], [3, 3]]
    ALLOWED_CONV_STRIDES = [1, 2,
                            (1, 1), (2, 2),
                            [1, 1], [2, 2]]

    POOLING_KERNEL = 2
    POOLING_STRIDE = 2
    POOLING_PADDING = 'valid'

    MIN_SHIFT = 0
    MAX_SHIFT = 12

    MAX_ACTIVATION_VALUES = {
        5: 31.0,
        10: 1023.0
    }
    RELU_CAP_10BITS = 31.96875

    def __init__(self, chip_id):
        self.chip_id = str(chip_id)

        global SPEC_INFO
        if SPEC_INFO is None:
            spec_path = os.path.join(os.path.dirname(__file__), 'specs', f'{chip_id}_spec.json')
            with open(os.path.realpath(spec_path), 'r') as f:
                info = json.load(f)
            SPEC_INFO = info
        self.info = SPEC_INFO

    def find_proper_device(self) -> DeviceInfo:
        for device in Specification.REGISTERED_DEVICES:
            if device.chip_id == self.chip_id:
                return device
        raise AttributeError('Invalid chip id: {}'.format(self.chip_id))

    def get_layer_limits(self):
        device = self.find_proper_device()
        return {
            'major': device.major_layer_limit,
            'sub': device.sub_layer_limit,
            'total': device.total_layer_limit
        }

    def get_info(self):
        return self.info

    def min_channels(self):
        return self.info['min_channels']

    def max_activation(self, bits=5):
        try:
            return Specification.MAX_ACTIVATION_VALUES[bits]
        except KeyError:
            raise ValueError('Invalid number of activation bits')

    def quantization_scheme(self, bit_mask):
        if bit_mask is None:
            valid_bit_masks = list(self.info['quantization'].keys())
            raise ValueError('Bit mask cannot be None. Available bit masks: {}'.format(", ".join(valid_bit_masks)))
        q = self.info['quantization'][str(bit_mask)]
        return q['weight'], q['bias']
