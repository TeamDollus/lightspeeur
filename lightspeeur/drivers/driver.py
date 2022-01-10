from lightspeeur.drivers.specification import Specification
from lightspeeur.drivers.utils import permute_axis

import os
import json
import ctypes
import platform
import logging

import numpy as np

SDK_VERSION_ENV_PARAM = os.environ.get('SDK_VERSION')
if SDK_VERSION_ENV_PARAM is not None:
    SDK_VERSION = int(SDK_VERSION_ENV_PARAM)
else:
    SDK_VERSION = 5

INT = ctypes.c_int
CHAR_ARRAY = ctypes.c_char_p
UNSIGNED_LONG = ctypes.c_ulong
UNSIGNED_LONG_LONG = ctypes.c_ulonglong
FLOAT = ctypes.c_float
BYTE = ctypes.c_uint8
VOID_ARRAY = ctypes.c_void_p
BOOL = ctypes.c_bool

BIT_SIZE = ctypes.sizeof(VOID_ARRAY)

logger = logging.getLogger('lightspeeur')
logger.setLevel(logging.INFO)
logging.basicConfig()


def instance_signature(cls, args=None, return_type=None):
    if args is not None:
        cls.argtypes = args
    if return_type is not None:
        cls.restype = return_type
    return cls


class Tensor(ctypes.Structure):
    pass


TENSOR_FIELDS = [
    [('width', INT), ('height', INT), ('depth', INT), ('stride', INT),
     ('buffer', VOID_ARRAY), ('customerBuffer', VOID_ARRAY), ('size', INT), ('format', INT),
     ('tag', VOID_ARRAY), ('next', ctypes.POINTER(Tensor)),

     ('width', INT), ('height', INT), ('depth', INT), ('stride', INT),
     ('buffer', VOID_ARRAY), ('size', INT), ('format', INT)]
]
Tensor._fields_ = TENSOR_FIELDS[0] if SDK_VERSION >= 5 else TENSOR_FIELDS[1]


class Driver:

    def __init__(self, library_path='bin/libGTILibrary.so', model_tools_path='bin/libmodeltools.so'):
        self.library_path = library_path
        self.model_tools_path = model_tools_path

    def __enter__(self):
        logger.info("Preparing libraries with SDK {}".format(SDK_VERSION))
        self.library = ctypes.CDLL(self.library_path)
        self.model_tools = ctypes.CDLL(self.model_tools_path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.library = None
        self.model_tools = None

    def compose(self, json_path: str, model_path: str):
        cls = instance_signature(self.model_tools.GtiComposeModelFile, [CHAR_ARRAY, CHAR_ARRAY])
        return cls(json_path.encode('ascii'), model_path.encode('ascii'))


class Model:

    def __init__(self, driver: Driver, chip_id: str, model_path: str):
        if platform.system() != 'Linux':
            raise NotImplementedError('Windows and macOS are not supported')

        if not os.path.exists(model_path):
            raise FileNotFoundError('Model does not exist in:', model_path)

        self.specification = Specification(chip_id)
        self.driver = driver
        self.model_path = model_path

    def __enter__(self):
        self.driver.__enter__()
        cls = instance_signature(self.driver.library.GtiCreateModel, [CHAR_ARRAY], UNSIGNED_LONG_LONG)

        self.instance = cls(self.model_path.encode('ascii'))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.instance is not None:
            cls = instance_signature(self.driver.library.GtiDestroyModel, [UNSIGNED_LONG_LONG], INT)
            cls(self.instance)

            self.instance = None
            self.driver.__exit__(exc_type, exc_val, exc_tb)

    def evaluate_internally(self, buffer: np.ndarray):
        if len(buffer.shape) == 4:
            buffer = buffer.squeeze(axis=0)
        elif len(buffer.shape) != 3:
            raise ValueError('Input dimension must be (height x width x channel) or (1 x height x width x channel)')

        height, width, channels = buffer.shape
        buffer = np.vstack(np.dsplit(buffer, channels))
        if SDK_VERSION >= 5:
            args = [width, height, channels, 0, buffer.ctypes.data, None, channels * height * width, 0, None, None]
        else:
            args = [width, height, channels, 0, buffer.ctypes.data, channels * height * width, 0]
        tensor = Tensor(*args)

        cls = instance_signature(self.driver.library.GtiEvaluate,
                                 [UNSIGNED_LONG_LONG, ctypes.POINTER(Tensor)],
                                 ctypes.POINTER(Tensor))

        return cls(self.instance, ctypes.byref(tensor))

    def evaluate(self, buffer: np.ndarray, activation_bits=5):
        tensor = self.evaluate_internally(buffer).contents
        shape = (1, tensor.depth, tensor.height, tensor.width)
        if SDK_VERSION >= 5:
            ptype = ctypes.POINTER(FLOAT) if activation_bits > 5 else ctypes.POINTER(BYTE)
        else:
            ptype = ctypes.POINTER(FLOAT)
        buf = ctypes.cast(tensor.buffer, ptype)
        res = (
            np.ctypeslib.as_array(buf, shape=(np.prod(shape),))
            .reshape(shape)
            .transpose(permute_axis('NCHW', 'NHWC'))
            .astype(np.float32)
        )
        if activation_bits == 5:
            return res
        return res / (self.specification.max_activation(activation_bits) / Specification.RELU_CAP_10BITS)

    def infer(self, buffer: np.ndarray):
        if SDK_VERSION >= 5:
            raise NotImplementedError('Inferring model in Python is not supported in SDK 5+. Use C/C++ API or SDK 4')
        tensor = self.evaluate_internally(buffer).contents
        body = ctypes.cast(tensor.buffer, CHAR_ARRAY).value
        return json.loads(body.decode('utf-8'))

    def version(self):
        cls = instance_signature(self.driver.library.GtiGetSDKVersion, return_type=CHAR_ARRAY)
        return cls()

    def evaluate_image(self, image, height, width, depth):
        if BIT_SIZE == 4:
            # 32-bit machine
            cls = instance_signature(self.driver.library.GtiImageEvaluate,
                                     [UNSIGNED_LONG, CHAR_ARRAY, INT, INT, INT],
                                     CHAR_ARRAY)
        else:
            # 64-bit machine
            cls = instance_signature(self.driver.library.GtiImageEvaluate,
                                     [UNSIGNED_LONG_LONG, CHAR_ARRAY, INT, INT, INT],
                                     CHAR_ARRAY)

        return cls(self.instance, image, height, width, depth)
