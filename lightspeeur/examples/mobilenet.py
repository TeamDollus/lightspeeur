import tensorflow as tf

from collections import namedtuple
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense as KerasDense, Flatten, BatchNormalization
from lightspeeur.layers import DepthwiseConv2D, Conv2D, ReLU
from lightspeeur.models import ModelStageAdvisor, LearningStage


train_datagen = ImageDataGenerator(rescale=1 / 255.)
valid_datagen = ImageDataGenerator(rescale=1 / 255.)

train_iter = train_datagen.flow_from_directory('dataset/train', target_size=(224, 224), batch_size=64)
valid_iter = valid_datagen.flow_from_directory('dataset/valid', target_size=(224, 224), batch_size=64)

chip_id = 2803
DepthwiseConvBlock = namedtuple("DepthwiseConvBlock", ["name", "stride", "in_channels", "out_channels"])
MOBILENET_DEPTHWISE_CONV_LAYERS = [
    DepthwiseConvBlock(name="conv2_1", stride=1, in_channels=32, out_channels=64),
    DepthwiseConvBlock(name="conv3_1", stride=2, in_channels=64, out_channels=128),
    DepthwiseConvBlock(name="conv3_2", stride=1, in_channels=128, out_channels=128),
    DepthwiseConvBlock(name="conv4_1", stride=2, in_channels=128, out_channels=256),
    DepthwiseConvBlock(name="conv4_2", stride=1, in_channels=256, out_channels=256),
    DepthwiseConvBlock(name="conv5_1", stride=2, in_channels=256, out_channels=512),
    DepthwiseConvBlock(name="conv5_2", stride=1, in_channels=512, out_channels=512),
    DepthwiseConvBlock(name="conv5_3", stride=1, in_channels=512, out_channels=512),
    DepthwiseConvBlock(name="conv5_4", stride=1, in_channels=512, out_channels=512),
    DepthwiseConvBlock(name="conv5_5", stride=1, in_channels=512, out_channels=512),
    DepthwiseConvBlock(name="conv5_6", stride=1, in_channels=512, out_channels=512),
    DepthwiseConvBlock(name="conv6_1", stride=2, in_channels=512, out_channels=1024),
    DepthwiseConvBlock(name="conv6_2", stride=1, in_channels=1024, out_channels=1024),
]


def conv_block(inputs, units, kernel_size=(1, 1), strides=(1, 1), name=None):
    outputs = Conv2D(units,
                     kernel_size,
                     chip_id,
                     strides=strides,
                     bit_mask=12,
                     use_bias=False,
                     name='{}/{}'.format(name, 'conv'))(inputs)
    outputs = BatchNormalization(name='{}/{}'.format(name, 'batch_norm'))(outputs)
    outputs = ReLU(chip_id, name=name)(outputs)
    return outputs


def depthwise_conv_block(inputs, strides=(1, 1), name=None):
    outputs = DepthwiseConv2D(chip_id,
                              strides=strides,
                              bit_mask=12,
                              use_bias=False,
                              name='{}/{}'.format(name, 'conv'))(inputs)
    outputs = BatchNormalization(name='{}/{}'.format(name, 'batch_norm'))(outputs)
    outputs = ReLU(chip_id, name=name)(outputs)
    return outputs


def build_mobilenet(inputs):
    outputs = conv_block(inputs, 32, kernel_size=(3, 3), strides=(2, 2), name='conv1_1')
    for layer in MOBILENET_DEPTHWISE_CONV_LAYERS:
        outputs = depthwise_conv_block(outputs, (layer.stride, layer.stride), name='{}_dw'.format(layer.name))
        outputs = conv_block(outputs, layer.out_channels, name='{}_pw'.format(layer.name))
    outputs = GlobalAveragePooling2D()(outputs)
    outputs = Flatten()(outputs)
    outputs = KerasDense(1024, activation='relu')(outputs)
    outputs = KerasDense(5, activation='softmax')(outputs)

    return Model(name='MobileNetV1', inputs=inputs, outputs=outputs)


custom_objects = {
    'Conv2D': Conv2D,
    'DepthwiseConv2D': DepthwiseConv2D,
    'ReLU': ReLU
}
model = tf.keras.models.load_model('mobilenet_transferred.h5', custom_objects=custom_objects)

compile_options = {
    'optimizer': 'adam',
    'loss': 'categorical_crossentropy',
    'metrics': ['accuracy']
}

advisor = ModelStageAdvisor(chip_id, model, compile_options)
while True:
    res = advisor.advance_stage()
    if res:
        if advisor.current_stage == LearningStage.MODEL_FUSION:
            epochs = 200
        else:
            epochs = 10
        advisor.propose(train_iter,
                        epochs=epochs,
                        validation_data=valid_iter,
                        steps_per_epoch=train_iter.samples // train_iter.batch_size)
    else:
        break

advisor.get_model().save('lightspeeur_mobilenet.h5')
