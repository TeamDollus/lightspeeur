import random
import time

from tensorflow.keras import datasets, layers as l, Model
from tensorflow.keras.models import load_model
from lightspeeur.layers import Conv2D, MaxPooling2D, ReLU, DepthwiseConv2D
from lightspeeur.models import ModelStageAdvisor, ModelConverter
from lightspeeur.drivers import Driver, Model as LightspeeurModel


def current_milliseconds():
    return round(time.time() * 1000)


(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

train_images, test_images = train_images / 255., test_images / 255.

chip_id = '2803'
train_model = False

if train_model:
    inputs = l.Input(shape=(28, 28, 1))

    x = Conv2D(32, (3, 3), chip_id, bit_mask=12, quantize=False, name='conv1/conv')(inputs)
    x = l.BatchNormalization(name='conv1/batch_norm')(x)
    x = ReLU(chip_id, quantize=False, name='conv1/relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (3, 3), chip_id, bit_mask=12, quantize=False, name='conv2/conv')(x)
    x = ReLU(chip_id, quantize=False, name='conv2/relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(32, (3, 3), chip_id, bit_mask=12, quantize=False, name='conv3/conv')(x)
    x = l.BatchNormalization(name='conv3/batch_norm')(x)
    x = ReLU(chip_id, quantize=False, name='conv3/relu')(x)
    x = MaxPooling2D()(x)

    x = l.Flatten()(x)
    x = l.Dense(64, activation='relu')(x)
    x = l.Dense(10, activation='softmax')(x)
    outputs = x
    model = Model(inputs=inputs, outputs=outputs, name='MNIST_CNN')

    compile_options = {
        'optimizer': 'adam',
        'loss': 'sparse_categorical_crossentropy',
        'metrics': ['accuracy']
    }
    advisor = ModelStageAdvisor(chip_id=chip_id, model=model, compile_options=compile_options)

    while True:
        res = advisor.advance_stage()
        if res:
            advisor.propose(train_images, train_labels, epochs=1, validation_split=0.2, batch_size=64)
        else:
            break

    advisor.get_model().save('lightspeeur_mnist.h5')
    model = advisor.get_model()
else:
    custom_objects = {
        'Conv2D': Conv2D,
        'DepthwiseConv2D': DepthwiseConv2D,
        'ReLU': ReLU
    }
    model = load_model('lightspeeur_mnist.h5', custom_objects=custom_objects)

# Conversion

now = current_milliseconds()
graph_name = 'conv'

driver = Driver()
converter = ModelConverter(chip_id=chip_id,
                           model=model,
                           graph={
                               graph_name: [['conv1/conv', 'conv1/relu'],
                                            ['conv2/conv', 'conv2/relu'],
                                            ['conv3/conv', 'conv3/relu']]
                           },
                           config_path='bin/libgticonfig2803.so',
                           debug=True)

results = converter.convert(driver)
result = results[0]
print('Time elapsed for conversion: {}ms'.format(current_milliseconds() - now))

# Evaluate

now = current_milliseconds()
driver = Driver()
lightspeeur_model = LightspeeurModel(driver, chip_id, result)
samples = 5

with lightspeeur_model:
    # lightspeeur model uses __enter__() to load a libraries
    print('Time elapsed for loading a model: {}ms'.format(current_milliseconds() - now))

    for _ in range(samples):
        now = current_milliseconds()
        sample_index = random.randint(0, test_images.shape[0])  # 0 ~ 10000

        sample_image = test_images[sample_index]
        sample_label = test_labels[sample_index]

        res = lightspeeur_model.evaluate(sample_image)
        print()
        print('Time elapsed for evaluate an image: {}ms'.format(current_milliseconds() - now))
        print('Result shape: {}'.format(res.shape))

