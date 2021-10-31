# Copyright (c) 2021 Project Bee4Exp.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from keras.layers import Activation, Conv2D, MaxPooling2D, Input, BatchNormalization, UpSampling2D, Add
from keras.models import Model


def SegmentationModel(input_shape=(256, 256, 3), out_channels=1):
    """ Creates segmentation CNN model.

    Args:
      input_shape:
        Shape of input without batch dimension.
      out_channels:
        Number of output channels in prediction.

    Returns:
      Keras model.
    """
    img_input = Input(shape=input_shape, dtype='float32')
    conv1 = Conv2D(64, (3, 3), activation='linear', padding='same', use_bias=False, name='conv1')(img_input)
    x = BatchNormalization(axis=3)(conv1)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    conv2 = Conv2D(128, (3, 3), activation='linear', padding='same', use_bias=False, name='conv2')(x)
    x = BatchNormalization(axis=3)(conv2)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    conv3 = Conv2D(256, (3, 3), activation='linear', padding='same', use_bias=False, name='conv3')(x)
    x = BatchNormalization(axis=3)(conv3)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    x = Conv2D(512, (1, 1), activation='linear', padding='same', use_bias=False, name='fc-conv2')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = UpSampling2D(2, interpolation='nearest')(x)
    x = Conv2D(256, (3, 3), activation='linear', padding='same', use_bias=False, name='conv3u')(x)
    x = Add()([x, conv3])
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = UpSampling2D(2, interpolation='nearest')(x)
    x = Conv2D(128, (3, 3), activation='linear', padding='same', use_bias=False, name='conv2u')(x)
    x = Add()([x, conv2])
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(out_channels, (1, 1), activation='linear', padding='same', use_bias=True, name='detector')(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs=img_input, outputs=x)

    return model
