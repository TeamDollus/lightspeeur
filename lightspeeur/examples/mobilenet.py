import os
import time
import random

import numpy as np
import tensorflow as tf

from collections import namedtuple
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense as KerasDense, Flatten, BatchNormalization, Input
from lightspeeur.layers import DepthwiseConv2D, Conv2D, ReLU
from lightspeeur.models import ModelStageAdvisor, ModelConverter
from lightspeeur.drivers import Driver, Model as LightspeeurModel, Specification

SDK_VERSION_ENV_PARAM = os.environ.get('SDK_VERSION')
if SDK_VERSION_ENV_PARAM is not None:
    SDK_VERSION = int(SDK_VERSION_ENV_PARAM)
else:
    SDK_VERSION = 5

if SDK_VERSION >= 5:
    library_path = 'bin/libGTILibrary.so'
else:
    library_path = 'bin/libGTILibrary.so.4.5.1.0.5795ef42'

chip_id = '2803'
spec = Specification(chip_id)


def current_milliseconds():
    return round(time.time() * 1000)


transfer_model = True
train_model = True
convert_model = True
evaluate_model = False


def truncate_for_chip(img):
    # [0, 255] -> [0, 31]
    # truncate: ((x >> 2) + 1) >> 1
    img = tf.cast(img, tf.uint8)
    img = tf.bitwise.right_shift(img, tf.ones_like(img) * 2)
    img = img + 1
    img = tf.bitwise.right_shift(img, tf.ones_like(img))
    img = tf.cast(img, tf.float32)
    img = tf.clip_by_value(img, 0, spec.max_activation())
    return img


train_datagen = ImageDataGenerator(preprocessing_function=truncate_for_chip)
valid_datagen = ImageDataGenerator(preprocessing_function=truncate_for_chip)

train_iter = train_datagen.flow_from_directory('dataset/train', target_size=(224, 224), batch_size=64)
valid_iter = valid_datagen.flow_from_directory('dataset/valid', target_size=(224, 224), batch_size=64)

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
                            steps_per_epoch=train_iter.samples // train_iter.batch_size,
                            clip_bias=True)
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
    model = tf.keras.models.load_model('lightspeeur_mobilenet.h5', custom_objects=custom_objects)


# Conversion
if convert_model:
    now = current_milliseconds()
    graph_name = 'mobilenet'

    last_image_size = None
    graph = []
    chunk = []
    for layer in model.layers:
        if isinstance(layer, Conv2D) or isinstance(layer, DepthwiseConv2D):
            current_image_size = layer.input.shape[1]
            if last_image_size is None:
                chunk.append(layer.name)
                last_image_size = current_image_size
                continue

            if last_image_size != current_image_size:
                graph.append(chunk)
                chunk = []

            last_image_size = current_image_size
            chunk.append(layer.name)
    graph.append(chunk)

    driver = Driver(library_path=library_path)
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
else:
    result = 'mobilenet_target/mobilenet.model'


# Evaluate
if evaluate_model:
    off_chip_model = Model(inputs=model.layers[-4].input,
                           outputs=model.layers[-1].output,
                           name='MobileNetV1OffChip')

    now = current_milliseconds()
    driver = Driver(library_path=library_path)
    lightspeeur_model = LightspeeurModel(driver, chip_id, result)
    samples = 5

    batch_images, batch_labels = valid_iter[0]
    classes = {v: k for k, v in valid_iter.class_indices.items()}

    with lightspeeur_model:
        print('Time elapsed for loading a model: {}ms'.format(current_milliseconds() - now))

        for _ in range(samples):
            now = current_milliseconds()
            sample_index = random.randint(0, len(batch_images))

            sample_image = batch_images[sample_index]
            sample_label = batch_labels[sample_index]

            r, g, b = tf.split(sample_image, num_or_size_splits=3, axis=2)
            sample_image = tf.concat([b, g, r], axis=2)
            sample_image = tf.cast(sample_image, tf.uint8)
            res = lightspeeur_model.evaluate(sample_image)
            print()
            print('Time elapsed for evaluate an image: {}ms'.format(current_milliseconds() - now))

            now = current_milliseconds()
            model_result = off_chip_model(res)
            print()

            predicted_index = np.argmax(model_result[0])
            ground_truth_index = np.argmax(sample_label)
            print('Predicated: \t{} ({} confidence)'.format(classes[predicted_index], model_result[0][predicted_index]))
            print('Ground-truth: \t{}'.format(classes[ground_truth_index]))
            print('Time elapsed for feed-forward off-chip model: {}ms'.format(current_milliseconds() - now))
