from __future__ import absolute_import
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D


def channels():
    return 'channels_first'


def layers(input_shape):
    channels_format = channels()
    return [
        ZeroPadding2D((2, 2), input_shape=input_shape, data_format=channels_format),
        Conv2D(64, (5, 5), padding='valid', data_format=channels_format),
        Activation('relu'),

        ZeroPadding2D((2, 2), data_format=channels_format),
        Conv2D(64, (5, 5), data_format=channels_format),
        Activation('relu'),

        ZeroPadding2D((1, 1), data_format=channels_format),
        Conv2D(64, (3, 3), data_format=channels_format),
        Activation('relu'),

        ZeroPadding2D((1, 1), data_format=channels_format),
        Conv2D(64, (3, 3), data_format=channels_format),
        Activation('relu'),

        ZeroPadding2D((1, 1), data_format=channels_format),
        Conv2D(64, (3, 3), data_format=channels_format),
        Activation('relu'),

        Flatten(),
        Dense(512),
        Activation('relu'),
    ]
