from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import tensorflow as tf
from sspcab.sspcab_tf import SSPCAB

import os

from keras.optimizers import Adam

tf.config.run_functions_eagerly(True)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.pop('TF_CONFIG', None)


model = Sequential()

# Encoder
"""
Each encoder is composed of three convolutional (conv)
layers, each followed by a max-pooling layer with a filter
size of 2 × 2 applied at a stride of 2. The conv layers are
formed of 3 × 3 filters. Each conv layer is followed by
Rectified Linear Units (ReLU) [70] as the activation function.
The first two conv layers consist of 32 filters, while the
third layer consists of 16 filters. The latent representation
is composed of 16 activation maps of size 8 × 8.
"""

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)))
model.add(MaxPooling2D((2, 2), padding='same', strides=2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same', strides=2))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))

# Decoder
"""
Each decoder starts with an upsampling layer, increasing
the spatial support of the activation maps by a factor of
2×. The upsampling operation is based on nearest neighbor
interpolation. After upsampling, we apply a conv layer with
16 filters of 3 × 3. The first upsampling and conv block
is followed by another two upsampling and conv blocks.
The last conv layer of an appearance decoder is formed of
a single conv filter, while the last conv layer of a motion
decoder is formed of two filters. In both cases, the number
of filters in the last conv layer is chosen such that the size of
the output matches the size of the input
"""
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(SSPCAB(name='sspcab', filters=64, kernel_dim=1, dilation=1, reduction_ratio=8))
model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae'])