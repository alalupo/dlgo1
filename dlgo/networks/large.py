from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D


def layers(input_shape):
    channels_format = 'channels_last'
    return [
        ZeroPadding2D((3, 3), input_shape=input_shape, data_format=channels_format),
        Conv2D(64, (7, 7), padding='valid', data_format=channels_format),
        Activation('relu'),

        ZeroPadding2D((2, 2), data_format=channels_format),
        Conv2D(64, (5, 5), data_format=channels_format),
        Activation('relu'),

        ZeroPadding2D((2, 2), data_format=channels_format),
        Conv2D(64, (5, 5), data_format=channels_format),
        Activation('relu'),

        ZeroPadding2D((2, 2), data_format=channels_format),
        Conv2D(48, (5, 5), data_format=channels_format),
        Activation('relu'),

        ZeroPadding2D((2, 2), data_format=channels_format),
        Conv2D(48, (5, 5), data_format=channels_format),
        Activation('relu'),

        ZeroPadding2D((2, 2), data_format=channels_format),
        Conv2D(32, (5, 5), data_format=channels_format),
        Activation('relu'),

        ZeroPadding2D((2, 2), data_format=channels_format),
        Conv2D(32, (5, 5), data_format=channels_format),
        Activation('relu'),

        Flatten(),
        Dense(1024),
        Activation('relu'),
    ]
