import tensorflow as tf
from keras.layers import Conv2D, Dense, Activation, Dropout, Flatten, Input
from keras.layers import ZeroPadding2D, MaxPooling2D

tf.get_logger().setLevel('WARNING')


def show_data_format():
    print(f'Image data format: {tf.keras.backend.image_data_format()}')


class AgentCriticNetwork:
    def __init__(self, encoder):
        self.encoder = encoder
        self.board_input = Input(shape=encoder.shape_for_keras(), name='board_input')
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
        self.board_input = Input(shape=encoder.shape_for_keras(), name='board_input')  # (None, 19, 19, 11)
        self.name = 'trainer'
        self.output = self.define_layers()

    def define_layers(self):
        starting = Dense(512, activation='relu')(self.board_input)
        net = Conv2D(48, (3, 3), activation='relu', padding='same')(starting)
        net = Dropout(rate=0.1)(net)

        net = Conv2D(48, (3, 3), padding='same', activation='relu')(net)
        net = MaxPooling2D(pool_size=(2, 2))(net)
        net = Dropout(rate=0.1)(net)

        net = Conv2D(48, (3, 3), padding='same', activation='relu')(net)
        net = MaxPooling2D(pool_size=(2, 2))(net)
        net = Dropout(rate=0.1)(net)

        net = Conv2D(48, (3, 3), padding='same', activation='relu')(net)
        net = MaxPooling2D(pool_size=(2, 2))(net)
        net = Dropout(rate=0.1)(net)

        net = ZeroPadding2D((2, 2))(net)
        net = Conv2D(48, (5, 5))(net)
        net = Activation('relu')(net)

        flat = Flatten()(net)
        output = Dense(self.num_classes, activation='softmax')(flat)

        return output
