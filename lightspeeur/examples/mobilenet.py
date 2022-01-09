import time
import random

import tensorflow as tf

from collections import namedtuple
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense as KerasDense, Flatten, BatchNormalization, Input
from lightspeeur.layers import DepthwiseConv2D, Conv2D, ReLU
from lightspeeur.models import ModelStageAdvisor, ModelConverter
from lightspeeur.drivers import Driver, Model as LightspeeurModel


def current_milliseconds():
    return round(time.time() * 1000)


transfer_model = True
train_model = True
evaluate_model = False

train_datagen = ImageDataGenerator(rescale=1 / 255.)
valid_datagen = ImageDataGenerator(rescale=1 / 255.)

train_iter = train_datagen.flow_from_directory('dataset/train', target_size=(224, 224), batch_size=64)
valid_iter = valid_datagen.flow_from_directory('dataset/valid', target_size=(224, 224), batch_size=64)

chip_id = '2803'
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


# Transfer Learning
if transfer_model:
    custom_objects = {
        'Conv2D': Conv2D,
        'DepthwiseConv2D': DepthwiseConv2D,
        'ReLU': ReLU
    }
    model = tf.keras.models.load_model('mobilenet_transferred.h5', custom_objects=custom_objects)
else:
    model = build_mobilenet(Input(shape=(224, 224, 3)))


# Train the model
if train_model:
    compile_options = {
        'optimizer': 'adam',
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy']
    }

    advisor = ModelStageAdvisor(chip_id, model, compile_options)
    while True:
        res = advisor.advance_stage()
        if res:
            advisor.propose(train_iter,
                            epochs=10,
                            validation_data=valid_iter,
                            steps_per_epoch=train_iter.samples // train_iter.batch_size)
        else:
            break

    advisor.get_model().save('lightspeeur_mobilenet.h5')
    model = advisor.get_model()
else:
    custom_objects = {
        'Conv2D': Conv2D,
        'DepthwiseConv2D': DepthwiseConv2D,
        'ReLU': ReLU
    }
    model = tf.keras.models.load_model('lightspeeur_mnist.h5', custom_objects=custom_objects)


# Conversion
now = current_milliseconds()
graph_name = 'mobilenet'

graph = [['conv1_1/conv', 'conv1_1']]
for conv in MOBILENET_DEPTHWISE_CONV_LAYERS:
    conv_name_depthwise = '{}_dw'.format(conv.name)
    conv_name_pointwise = '{}_pw'.format(conv.name)
    graph.append(['{}/conv'.format(conv_name_depthwise), conv_name_depthwise])
    graph.append(['{}/conv'.format(conv_name_depthwise), conv_name_pointwise])

driver = Driver()
converter = ModelConverter(chip_id=chip_id,
                           model=model,
                           graph={
                               graph_name: graph
                           },
                           config_path='bin/libgticonfig2803.so',
                           debug=True)

results = converter.convert(driver)
result = results[0]
print('Time elapsed for conversion: {}ms'.format(current_milliseconds() - now))


# Evaluate
if evaluate_model:
    now = current_milliseconds()
    driver = Driver()
    lightspeeur_model = LightspeeurModel(driver, chip_id, result)
    samples = 5

    batch_images, batch_labels = valid_iter[0]

    with lightspeeur_model:
        print('Time elapsed for loading a model: {}ms'.format(current_milliseconds() - now))

        for _ in range(samples):
            now = current_milliseconds()
            sample_index = random.randint(0, len(batch_images))

            sample_image = batch_images[sample_index]
            sample_label = batch_labels[sample_index]

            res = lightspeeur_model.evaluate(sample_image)
            print()
            print('Time elapsed for evaluate an image: {}ms'.format(current_milliseconds() - now))
            print('Result shape: {}'.format(res.shape))
