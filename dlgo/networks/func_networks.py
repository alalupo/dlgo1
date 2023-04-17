import tensorflow as tf
from keras.layers import Conv2D, Dense, Activation, Dropout, Flatten, Input
from keras.layers import ZeroPadding2D, MaxPooling2D

tf.get_logger().setLevel('WARNING')


def show_data_format():
    print(f'Image data format: {tf.keras.backend.image_data_format()}')


class AgentCriticNetwork:
    def __init__(self, encoder):
        self.encoder = encoder
        self.board_input = Input(shape=encoder.shape_for_others(), name='board_input')
        self.policy_output, self.value_output = self.define_layers()
        self.name = 'agent-critic'

    def define_layers(self):
        conv1a = ZeroPadding2D((2, 2))(self.board_input)
        conv1b = Conv2D(64, (5, 5), activation='relu')(conv1a)

        conv2a = ZeroPadding2D((1, 1))(conv1b)
        conv2b = Conv2D(64, (3, 3), activation='relu')(conv2a)

        flat = Flatten()(conv2b)
        processed_board = Dense(512)(flat)

        policy_hidden_layer = Dense(512, activation='relu')(processed_board)
        policy_output = Dense(self.encoder.num_points(), activation='softmax')(policy_hidden_layer)
        value_hidden_layer = Dense(512, activation='relu')(processed_board)
        value_output = Dense(1, activation='tanh')(value_hidden_layer)
        return policy_output, value_output


class TrainerNetwork:
    def __init__(self, encoder):
        self.encoder = encoder
        self.num_classes = self.encoder.num_points()
        self.board_input = Input(shape=encoder.shape_for_others(), name='board_input')  # (None, 19, 19, 11)
        self.name = 'trainer'
        self.output = self.define_layers()

    def define_layers(self):
        # conv1a = ZeroPadding2D((3, 3))(self.board_input)
        # conv1b = Conv2D(64, (5, 5), activation='relu')(conv1a)
        # conv1c = Activation('relu')(conv1b)
        # conv1d = Dropout(rate=0.1)(conv1c)
        #
        # conv2a = ZeroPadding2D((2, 2))(conv1d)
        # conv2b = Conv2D(64, (3, 3), activation='relu')(conv2a)
        # conv2c = Dropout(rate=0.1)(conv2b)
        #
        # conv3a = ZeroPadding2D((2, 2))(conv2c)
        # conv3b = Conv2D(48, (2, 2), activation='relu')(conv3a)
        # conv3c = Activation('relu')(conv3b)
        # conv3d = Dropout(rate=0.1)(conv3c)
        #
        # conv4a = ZeroPadding2D((2, 2))(conv3d)
        # conv4b = Conv2D(48, (2, 2), activation='relu')(conv4a)
        # conv4c = Activation('relu')(conv4b)
        # conv4d = Dropout(rate=0.1)(conv4c)
        #
        # conv5a = ZeroPadding2D((2, 2))(conv4d)
        # conv5b = Conv2D(48, (2, 2), activation='relu')(conv5a)
        # conv5c = Activation('relu')(conv5b)
        # conv5d = Dropout(rate=0.1)(conv5c)
        #
        # flat = Flatten()(conv5d)
        # processed_board = Dense(512)(flat)
        #
        # output = Dense(self.num_classes, activation='softmax')(processed_board)

        starting = Dense(512, activation='relu')(self.board_input)
        conv1a = Conv2D(48,
                        kernel_size=(3, 3),
                        activation='relu',
                        padding='same')(starting)
        conv1b = Dropout(rate=0.1)(conv1a)

        conv2a = Conv2D(48, (3, 3),
                        padding='same', activation='relu')(conv1b)
        conv2b = MaxPooling2D(pool_size=(2, 2))(conv2a)
        conv2c = Dropout(rate=0.1)(conv2b)

        conv3a = Conv2D(48, (3, 3),
                        padding='same', activation='relu')(conv2c)
        conv3b = MaxPooling2D(pool_size=(2, 2))(conv3a)
        conv3c = Dropout(rate=0.1)(conv3b)

        conv4a = Conv2D(48, (3, 3),
                        padding='same', activation='relu')(conv3c)
        conv4b = MaxPooling2D(pool_size=(2, 2))(conv4a)
        conv4c = Dropout(rate=0.1)(conv4b)

        conv5a = ZeroPadding2D((2, 2))(conv4c)
        conv5b = Conv2D(48, (5, 5))(conv5a)
        conv5c = Activation('relu')(conv5b)

        flat = Flatten()(conv5c)
        output = Dense(self.num_classes, activation='softmax')(flat)

        return output
