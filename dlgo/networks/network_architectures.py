import tensorflow as tf

keras = tf.keras
from keras.layers import Conv2D, Dense, Activation, Dropout, Flatten, Input
from keras.layers import ZeroPadding2D, MaxPooling2D, BatchNormalization, LeakyReLU
from keras.layers import Add
from keras.regularizers import l2

from dlgo.zero import ZeroEncoder


# AlphaGoZero style network

class Network:
    def __init__(self, board_size):
        self.board_size = board_size
        self.encoder = ZeroEncoder(self.board_size)
        self.board_input = Input(shape=self.encoder.channels_last_shape(), name='board_input')
        self.num_filters = 256
        self.kernel_size = 3
        self.name = 'strong'
        self.policy_output, self.value_output = self.define_layers()

    def define_layers(self):

        pb = self.board_input
        # 4 conv layers with batch normalization
        for i in range(4):
            pb = Conv2D(64, (3, 3), padding='same')(pb)
            pb = BatchNormalization(axis=1)(pb)
            pb = Activation('relu')(pb)
        # Policy output
        policy_conv = Conv2D(2, (1, 1))(pb)
        policy_batch = BatchNormalization(axis=1)(policy_conv)
        policy_relu = Activation('relu')(policy_batch)
        policy_flat = Flatten()(policy_relu)
        policy_output = Dense(self.encoder.num_moves(), activation='softmax')(
            policy_flat)
        # Value output
        value_conv = Conv2D(1, (1, 1))(pb)
        value_batch = BatchNormalization(axis=1)(value_conv)
        value_relu = Activation('relu')(value_batch)
        value_flat = Flatten()(value_relu)
        value_hidden = Dense(256, activation='relu')(value_flat)
        value_output = Dense(1, activation='tanh')(value_hidden)
        return policy_output, value_output


# AlphaGo style networks

class StrongPolicyNetwork:
    def __init__(self, encoder):
        self.encoder = encoder
        self.num_classes = self.encoder.num_points()
        self.board_input = Input(shape=encoder.shape_for_keras(), name='board_input')
        self.num_filters = 256
        self.kernel_size = 3
        self.name = 'strong'
        self.output = self.define_layers()

    def define_layers(self):
        self.name = f'{self.name}_improved3'
        net = Conv2D(self.num_filters, self.kernel_size, padding='same', activation='relu')(self.board_input)
        net = BatchNormalization()(net)

        for i in range(4):
            skip = net
            net = Conv2D(self.num_filters, self.kernel_size, padding='same', activation='relu')(net)
            net = BatchNormalization()(net)
            net = Conv2D(self.num_filters, self.kernel_size, padding='same')(net)
            net = BatchNormalization()(net)
            net = Add()([net, skip])
            net = Activation('relu')(net)

        net = Conv2D(filters=1, kernel_size=1, padding='same', activation='relu')(net)
        net = BatchNormalization()(net)
        net = Flatten()(net)
        output = Dense(self.num_classes, activation='softmax')(net)

        return output

    # self.name = f'{self.name}_improved'
    # net = Conv2D(self.num_filters, self.kernel_size, padding='same', activation='relu')(self.board_input)
    # net = BatchNormalization()(net)
    # for i in range(3):
    #     skip = net
    #     net = Conv2D(self.num_filters, self.kernel_size, padding='same', activation='relu')(net)
    #     net = BatchNormalization()(net)
    #     net = Conv2D(self.num_filters, self.kernel_size, padding='same')(net)
    #     net = BatchNormalization()(net)
    #     net = Add()([net, skip])
    #     net = Activation('relu')(net)
    # net = Conv2D(filters=1, kernel_size=1, padding='same', activation='softmax')(net)
    # output = Flatten()(net)
    # return output

    # # 6: 3xbefore skip (20% val_accuracy on 50,000 train set)
    # self.name = f'{self.name}6'
    # net = ZeroPadding2D(padding=3)(self.board_input)
    # net = Conv2D(48, (7, 7), kernel_regularizer=reg_lambda)(net)
    # net = BatchNormalization()(net)
    # net = LeakyReLU(alpha=0.1)(net)
    # net = ZeroPadding2D(padding=2)(net)
    # net = Conv2D(32, (5, 5), kernel_regularizer=reg_lambda)(net)
    # net = BatchNormalization()(net)
    # net = LeakyReLU(alpha=0.1)(net)
    # net = ZeroPadding2D(padding=2)(net)
    # net = Conv2D(32, (5, 5), kernel_regularizer=reg_lambda)(net)
    # net = BatchNormalization()(net)
    # net = LeakyReLU(alpha=0.1)(net)
    # skip = net
    # net = ZeroPadding2D(padding=2)(net)
    # net = Conv2D(32, (5, 5), kernel_regularizer=reg_lambda)(net)
    # net = BatchNormalization()(net)
    # net = LeakyReLU(alpha=0.1)(net)
    # net = ZeroPadding2D(padding=2)(net)
    # net = Conv2D(32, (5, 5), kernel_regularizer=reg_lambda)(net)
    # net = BatchNormalization()(net)
    # net = Add()([net, skip])
    # net = LeakyReLU(alpha=0.1)(net)
    # flat = Flatten()(net)
    # flat = Dropout(0.5)(flat)
    # output = Dense(self.num_classes, activation='softmax')(flat)
    # return output


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

        net = Conv2D(self.num_filters, self.first_kernel_size, padding='same', kernel_regularizer=reg_lambda)(
            self.board_input)
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
