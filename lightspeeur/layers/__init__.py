# Activation layers
from lightspeeur.layers.activation import ReLU

# Convolutional layers
from lightspeeur.layers.convolutional import Conv2D
from lightspeeur.layers.convolutional import Conv2DTranspose
from lightspeeur.layers.convolutional import DepthwiseConv2D

# Dense layers
from lightspeeur.layers.dense import Dense

# Pooling layers
from lightspeeur.layers.pooling import MaxPooling2D
from lightspeeur.layers.pooling import TopLeftPooling2D

# Quantizable layers
from lightspeeur.layers.quantization import QuantizableLayer, quantize_image

# Constraints
from lightspeeur.layers.constraints import ClippingBiasConstraint


custom_objects = {
    'ReLU': ReLU,
    'Conv2D': Conv2D,
    'Conv2DTranspose': Conv2DTranspose,
    'DepthwiseConv2D': DepthwiseConv2D,
    'MaxPooling2D': MaxPooling2D,
    'TopLeftPooling2D': TopLeftPooling2D
}
