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
        self.num_filters = 192
        self.first_kernel_size = 5
        self.other_kernel_size = 3
        self.name = 'strong_policy'
        self.output = self.define_layers(l2(0.01))

    def define_layers(self, reg_lambda):
        self.name = f'{self.name}_default'

        net = Conv2D(self.num_filters, self.first_kernel_size, padding='same', activation='relu')(self.board_input)

        net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
        net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
        net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
        net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
        net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
        net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
        net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
        net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
        net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
        net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)

        net = Conv2D(filters=1, kernel_size=1, padding='same', activation='softmax')(net)
        output = Flatten()(net)
        return output


class ValueNetwork:
    def __init__(self, encoder):
        self.encoder = encoder
        self.board_input = Input(shape=encoder.shape_for_keras(), name='board_input')
        self.num_filters = 192
        self.first_kernel_size = 5
        self.other_kernel_size = 3
        self.name = 'value'
        self.output = self.define_layers(l2(0.01))

    def define_layers(self, reg_lambda):

        self.name = f'{self.name}_default'

        net = Conv2D(self.num_filters, self.first_kernel_size, padding='same', activation='relu')(self.board_input)

        net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
        net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
        net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
        net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
        net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
        net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
        net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
        net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
        net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)
        net = Conv2D(self.num_filters, self.other_kernel_size, padding='same', activation='relu')(net)

        net = Conv2D(filters=1, kernel_size=1, padding='same', activation='relu')(net)
        flat = Flatten()(net)
        dense = Dense(256, activation='relu')(flat)
        output = Dense(1, activation='tanh')(dense)
        return output


class FastPolicyNetwork:
    def __init__(self, encoder):
        self.encoder = encoder
        self.board_input = Input(shape=encoder.shape_for_keras(), name='board_input')
        self.num_classes = self.encoder.num_points()
        self.num_filters = 192
        self.first_kernel_size = 5
        self.other_kernel_size = 3
        self.name = 'fast_policy'
        self.output = self.define_layers(l2(0.01))

    def define_layers(self, reg_lambda):
        self.name = f'{self.name}_default'

        net = ZeroPadding2D(padding=3)(self.board_input)
        net = Conv2D(48, (7, 7))(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = ZeroPadding2D(padding=2)(net)
        net = Conv2D(32, (5, 5))(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.1)(net)

        # Skip connection from layer 2 to layer 4
        skip = net

        net = ZeroPadding2D(padding=2)(net)
        net = Conv2D(32, (5, 5))(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = ZeroPadding2D(padding=2)(net)
        net = Conv2D(32, (5, 5))(net)
        net = BatchNormalization()(net)

        # Add skip connection to layer 4
        net = Add()([net, skip])

        net = LeakyReLU(alpha=0.1)(net)

        flat = Flatten()(net)
        flat = Dropout(0.5)(flat)
        output = Dense(self.num_classes, activation='softmax')(flat)

        return output