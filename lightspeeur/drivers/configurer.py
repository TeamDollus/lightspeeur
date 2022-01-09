import os
import ctypes
import json

from lightspeeur.drivers.driver import CHAR_ARRAY, BOOL, instance_signature


class Configurer:

    def __init__(self, name, config_path, kernels, biases, data_file, dst_data_file, debug=False):
        self.config_path = config_path
        self.kernels = kernels
        self.biases = biases
        self.data_file = data_file
        self.dst_data_file = dst_data_file
        self.debug = debug
        self.work_dir = os.path.join('{}_works'.format(name))
        self.work_file = os.path.join(self.work_dir, 'config')
        self.config_lib = None

    def __enter__(self):
        if not os.path.exists(self.config_path):
            raise ValueError('Config binary file is not existed at: {}'.format(self.config_path))
        self.config_lib = ctypes.CDLL(self.config_path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.config_lib is not None:
            del self.config_lib

    def configure(self):
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir, exist_ok=True)

        with open(self.data_file, 'r') as f:
            body = json.load(f)

            models = body['model']
            for model in models:
                chip_id = model['ChipType']
                if self.config_lib is not None:
                    cls = instance_signature(self.config_lib.GtiConvertInternalToSDK,
                                             [CHAR_ARRAY, CHAR_ARRAY, CHAR_ARRAY, CHAR_ARRAY,
                                              CHAR_ARRAY, CHAR_ARRAY, CHAR_ARRAY, BOOL])
                    cls(self.data_file.encode('ascii'),
                        self.kernels.encode('ascii'),
                        self.biases.encode('ascii'),
                        'GTI{}'.format(chip_id).encode('ascii'),
                        self.dst_data_file.encode('ascii'),
                        self.work_file.encode('ascii'),
                        self.work_dir.encode('ascii'),
                        self.debug)
