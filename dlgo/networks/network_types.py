from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D, ZeroPadding2D


class SmallNetwork:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.channels_format = 'channels_last'
        self.name = 'small'

    def layers(self):
        return [
            ZeroPadding2D(padding=3, input_shape=self.input_shape, data_format=self.channels_format),  # <1>
            Conv2D(48, (7, 7), data_format=self.channels_format),
            Activation('relu'),

            Dropout(rate=0.25),

            ZeroPadding2D(padding=2, data_format=self.channels_format),  # <2>
            Conv2D(32, (5, 5), data_format=self.channels_format),
            Activation('relu'),

            Dropout(rate=0.25),

            ZeroPadding2D(padding=2, data_format=self.channels_format),
            Conv2D(32, (5, 5), data_format=self.channels_format),
            Activation('relu'),

            Dropout(rate=0.25),

            ZeroPadding2D(padding=2, data_format=self.channels_format),
            Conv2D(32, (5, 5), data_format=self.channels_format),
            Activation('relu'),

            Flatten(),
            Dense(512),
            Activation('relu'),
        ]


class MediumNetwork:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.channels_format = 'channels_last'
        self.name = 'medium'

    def layers(self):
        return [
            ZeroPadding2D((2, 2), input_shape=self.input_shape, data_format=self.channels_format),
            Conv2D(64, (5, 5), padding='valid', data_format=self.channels_format),
            Activation('relu'),

            Dropout(rate=0.25),

            ZeroPadding2D((2, 2), data_format=self.channels_format),
            Conv2D(64, (5, 5), data_format=self.channels_format),
            Activation('relu'),

            Dropout(rate=0.25),

            ZeroPadding2D((1, 1), data_format=self.channels_format),
            Conv2D(64, (3, 3), data_format=self.channels_format),
            Activation('relu'),

            Dropout(rate=0.5),

            ZeroPadding2D((1, 1), data_format=self.channels_format),
            Conv2D(64, (3, 3), data_format=self.channels_format),
            Activation('relu'),

            Dropout(rate=0.25),

            ZeroPadding2D((1, 1), data_format=self.channels_format),
            Conv2D(64, (3, 3), data_format=self.channels_format),
            Activation('relu'),

            Flatten(),
            Dense(512),
            Activation('relu'),
        ]


class LargeNetwork:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.channels_format = 'channels_last'
        self.name = 'large'

    def layers(self):
        return [
            ZeroPadding2D((3, 3), input_shape=self.input_shape, data_format=self.channels_format),
            Conv2D(64, (7, 7), padding='valid', data_format=self.channels_format),
            Activation('relu'),

            ZeroPadding2D((2, 2), data_format=self.channels_format),
            Conv2D(64, (5, 5), data_format=self.channels_format),
            Activation('relu'),

            ZeroPadding2D((2, 2), data_format=self.channels_format),
            Conv2D(64, (5, 5), data_format=self.channels_format),
            Activation('relu'),

            ZeroPadding2D((2, 2), data_format=self.channels_format),
            Conv2D(48, (5, 5), data_format=self.channels_format),
            Activation('relu'),

            ZeroPadding2D((2, 2), data_format=self.channels_format),
            Conv2D(48, (5, 5), data_format=self.channels_format),
            Activation('relu'),

            ZeroPadding2D((2, 2), data_format=self.channels_format),
            Conv2D(32, (5, 5), data_format=self.channels_format),
            Activation('relu'),

            ZeroPadding2D((2, 2), data_format=self.channels_format),
            Conv2D(32, (5, 5), data_format=self.channels_format),
            Activation('relu'),

            Flatten(),
            Dense(1024),
            Activation('relu'),
        ]
