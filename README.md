# Lightspeeur

Lightspeeur TensorFlow Model Development Framework

### Supported Layers

#### Conv2D, Conv2DTranspose and DepthwiseConv2D
```python
from lightspeeur.layers import Conv2D, Conv2DTranspose, DepthwiseConv2D
from lightspeeur.drivers.specification import UpSampleFillingMode

# Conv2D
outputs = Conv2D(32, (3, 3), 
                 strides=(1, 1),
                 chip_id='2803', 
                 bit_mask=12)(inputs)

# Conv2DTranspose
outputs = Conv2DTranspose(64, (3, 3), 
                          strides=(1, 1),
                          chip_id='2803', 
                          upsample_filling_mode=UpSampleFillingMode.REPEAT, 
                          bit_mask=12)(inputs)

# DepthwiseConv2D
outputs = DepthwiseConv2D(chip_id='2803',
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          bit_mask=12)(inputs)
```

#### ReLU
```python
from lightspeeur.layers import ReLU

outputs = ReLU(chip_id='2803')(inputs)
```

#### MaxPooling2D
```python
from lightspeeur.layers import MaxPooling2D

outputs = MaxPooling2D()(inputs)
```

### Example
```python
import tensorflow as tf

from tensorflow.keras import Model
from lightspeeur.models import ModelStageAdvisor
from lightspeeur.layers import Conv2D, ReLU, MaxPooling2D

l = tf.keras.layers

# Model Definition
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
model = Model(inputs=inputs, outputs=outputs, name='mnist_conv')

# Model Advisor
compile_options = {
  'optimizer': 'adam',
  'loss': 'sparse_categorical_crossentropy',
  'metrics': ['accuracy']
}
advisor = ModelStageAdvisor(chip_id=chip_id,
                            model=model,
                            compile_options=compile_options)
# Fit the Model
while True:
  advanced = advisor.advance_stage()
  if advanced:
    advisor.propose(train_x, train_y, epochs=10, validation_split=0.2)
  else:
    break

advisor.get_model().save('lightspeeur_model.hdf5')
```