from enum import Enum

import os
import json


class UpSampleFillingMode(Enum):
    REPEAT = 1
    ZERO = 2


SPEC_INFO = None


class Specification:

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

    LAYER_LIMITS = {
        '2803': {
            'major': 6,
            'sub': 12,
            'total': 72
        }
    }

    def __init__(self, chip_id):
        self.chip_id = str(chip_id)

        global SPEC_INFO
        if SPEC_INFO is None:
            spec_path = os.path.join(os.path.dirname(__file__), 'specs', f'{chip_id}_spec.json')
            with open(os.path.realpath(spec_path), 'r') as f:
                info = json.load(f)
            SPEC_INFO = info
        self.info = SPEC_INFO

    def get_layer_limits(self):
        if self.chip_id in Specification.LAYER_LIMITS:
            return Specification.LAYER_LIMITS[self.chip_id]
        return None

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
