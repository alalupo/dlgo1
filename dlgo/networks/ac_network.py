from keras.layers import Conv2D, Dense, Flatten, Input
from keras.layers import ZeroPadding2D


class AgentCriticNetwork:
    def __init__(self, encoder):
        self.encoder = encoder
        self.board_input = Input(shape=encoder.shape_for_others(), name='board_input')
        self.policy_output, self.value_output = self.define_layers()

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



