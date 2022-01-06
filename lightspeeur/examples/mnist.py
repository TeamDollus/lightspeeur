from tensorflow.keras import datasets, layers as l, Model
from lightspeeur.layers import Conv2D, MaxPooling2D, ReLU
from lightspeeur.models import ModelStageAdvisor


(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

train_images, test_images = train_images / 255., test_images / 255.

chip_id = '2803'
inputs = l.Input(shape=(28, 28, 1))
x = Conv2D(32, (3, 3), chip_id, bit_mask=12, quantize=False)(inputs)
x = l.BatchNormalization()(x)
x = ReLU(chip_id, quantize=False)(x)
x = MaxPooling2D()(x)
x = Conv2D(64, (3, 3), chip_id, bit_mask=12, quantize=False)(x)
x = ReLU(chip_id, quantize=False)(x)
x = MaxPooling2D()(x)
x = Conv2D(32, (3, 3), chip_id, bit_mask=12, quantize=False)(x)
x = l.BatchNormalization()(x)
x = ReLU(chip_id, quantize=False)(x)
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
