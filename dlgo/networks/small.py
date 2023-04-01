from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D


def channels():
    return 'channels_first'


def layers(input_shape):
    channels_format = channels()
    return [
        ZeroPadding2D(padding=3, input_shape=input_shape, data_format=channels_format),  # <1>
        Conv2D(48, (7, 7), data_format=channels_format),
        Activation('relu'),

        ZeroPadding2D(padding=2, data_format=channels_format),  # <2>
        Conv2D(32, (5, 5), data_format=channels_format),
        Activation('relu'),

        ZeroPadding2D(padding=2, data_format=channels_format),
        Conv2D(32, (5, 5), data_format=channels_format),
        Activation('relu'),

        ZeroPadding2D(padding=2, data_format=channels_format),
        Conv2D(32, (5, 5), data_format=channels_format),
        Activation('relu'),

        Flatten(),
        Dense(512),
        Activation('relu'),
    ]

