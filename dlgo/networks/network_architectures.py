import tensorflow as tf

keras = tf.keras
from keras.layers import Conv2D, Dense, Activation, Dropout, Flatten, Input
from keras.layers import ZeroPadding2D, MaxPooling2D, BatchNormalization, LeakyReLU
from keras.layers import Add
from keras.regularizers import l2


class StrongPolicyNetwork:
    def __init__(self, encoder):
        self.encoder = encoder
        self.board_input = Input(shape=encoder.shape_for_keras(), name='board_input')
        self.num_filters = 128
        self.kernel_size = 5
        self.name = 'strong'
        self.output = self.define_layers()

    def define_layers(self):
        self.name = f'{self.name}_improved'
        net = Conv2D(self.num_filters, self.kernel_size, padding='same', activation='relu')(self.board_input)
        net = BatchNormalization()(net)

        for i in range(3):
            skip = net
            net = Conv2D(self.num_filters, self.kernel_size, padding='same', activation='relu')(net)
            net = BatchNormalization()(net)
            net = Conv2D(self.num_filters, self.kernel_size, padding='same')(net)
            net = BatchNormalization()(net)
            net = Add()([net, skip])
            net = Activation('relu')(net)

        net = Conv2D(filters=1, kernel_size=1, padding='same', activation='softmax')(net)
        output = Flatten()(net)
        return output


# class StrongPolicyNetwork:
#     def __init__(self, encoder):
#         self.encoder = encoder
#         self.board_input = Input(shape=encoder.shape_for_keras(), name='board_input')
#         self.num_filters = 192
#         self.first_kernel_size = 5
#         self.other_kernel_size = 3
#         self.name = 'strong_policy'
#         self.output = self.define_layers(l2(0.01))
#
#     def define_layers(self, reg_lambda):
#         self.name = f'{self.name}_default'
#
#         net = Conv2D(self.num_filters, self.first_kernel_size, padding='same', activation='relu')(self.board_input)
#
#         net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
#         net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
#         net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
#         net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
#         net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
#         net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
#         net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
#         net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
#         net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
#         net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
#
#         net = Conv2D(filters=1, kernel_size=1, padding='same', activation='softmax')(net)
#         output = Flatten()(net)
#         return output

class ValueNetwork:
    def __init__(self, encoder):
        self.encoder = encoder
        self.board_input = Input(shape=encoder.shape_for_keras(), name='board_input')
        self.num_filters = 256
        self.first_kernel_size = 5
        self.other_kernel_size = 3
        self.residual_blocks = 19
        self.name = 'value'
        self.output = self.define_layers(l2(0.01))

    def define_layers(self, reg_lambda):

        self.name = f'{self.name}_improved'

        net = Conv2D(self.num_filters, self.first_kernel_size, padding='same', kernel_regularizer=reg_lambda)(self.board_input)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)

        for _ in range(self.residual_blocks):
            skip = net

            net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', kernel_regularizer=reg_lambda)(net)
            net = BatchNormalization()(net)
            net = Activation('relu')(net)

            net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', kernel_regularizer=reg_lambda)(net)
            net = BatchNormalization()(net)
            net = Add()([net, skip])
            net = Activation('relu')(net)

        net = Conv2D(filters=1, kernel_size=1, padding='same', kernel_regularizer=reg_lambda)(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)

        flat = Flatten()(net)
        dense = Dense(256, activation='relu')(flat)
        output = Dense(1, activation='tanh')(dense)

        return output


# class ValueNetwork:
#     def __init__(self, encoder):
#         self.encoder = encoder
#         self.board_input = Input(shape=encoder.shape_for_keras(), name='board_input')
#         self.num_filters = 192
#         self.first_kernel_size = 5
#         self.other_kernel_size = 3
#         self.name = 'value'
#         self.output = self.define_layers(l2(0.01))
#
#     def define_layers(self, reg_lambda):
#
#         self.name = f'{self.name}_default'
#
#         net = Conv2D(self.num_filters, self.first_kernel_size, padding='same', activation='relu')(self.board_input)
#
#         net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
#         net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
#         net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
#         net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
#         net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
#         net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
#         net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
#         net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
#         net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
#         net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
#
#         net = Conv2D(filters=1, kernel_size=1, padding='same', activation='relu')(net)
#         flat = Flatten()(net)
#         dense = Dense(256, activation='relu')(flat)
#         output = Dense(1, activation='tanh')(dense)
#         return output


class FastPolicyNetwork:
    def __init__(self, encoder):
        self.encoder = encoder
        self.board_input = Input(shape=encoder.shape_for_keras(), name='board_input')
        self.num_classes = self.encoder.num_points()
        self.num_filters = 64
        self.first_kernel_size = 5
        self.other_kernel_size = 3
        self.name = 'fast'
        self.output = self.define_layers(l2(0.01))

    def define_layers(self, reg_lambda):
        self.name = f'{self.name}_improved'

        net = ZeroPadding2D(padding=3)(self.board_input)
        net = Conv2D(16, (7, 7))(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = ZeroPadding2D(padding=2)(net)
        net = Conv2D(32, (5, 5))(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.1)(net)

        flat = Flatten()(net)
        flat = Dropout(0.5)(flat)
        output = Dense(self.num_classes, activation='softmax')(flat)

        return output
