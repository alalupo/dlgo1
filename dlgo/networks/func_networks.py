import tensorflow as tf
keras = tf.keras
from keras.layers import Conv2D, Dense, Activation, Dropout, Flatten, Input
from keras.layers import ZeroPadding2D, MaxPooling2D, BatchNormalization, LeakyReLU
from keras.layers import Add
from keras.regularizers import l2

tf.get_logger().setLevel('WARNING')


def show_data_format():
    print(f'Image data format: {tf.keras.backend.image_data_format()}')


class AgentCriticNetwork:
    def __init__(self, encoder):
        self.encoder = encoder
        self.board_input = Input(shape=encoder.shape_for_keras(), name='board_input')
        self.policy_output, self.value_output = self.define_layers()
        self.name = 'agent_critic'

    def define_layers(self):
        conv1a = ZeroPadding2D((2, 2), name='first_zero_pad')(self.board_input)
        conv1b = Conv2D(64, (5, 5), activation='relu', name='first_conv2D')(conv1a)

        conv2a = ZeroPadding2D((1, 1), name='second_zero_pad')(conv1b)
        conv2b = Conv2D(64, (3, 3), activation='relu', name='second_conv2D')(conv2a)

        flat = Flatten(name='flat')(conv2b)
        processed_board = Dense(512, name='processed_board')(flat)

        policy_hidden_layer = Dense(512, activation='relu', name='policy_hidden_layer')(processed_board)
        policy_output = Dense(self.encoder.num_points(), activation='softmax', name='policy_output')(policy_hidden_layer)
        value_hidden_layer = Dense(512, activation='relu', name='value_hidden_layer')(processed_board)
        value_output = Dense(1, activation='tanh', name='value_output')(value_hidden_layer)
        return policy_output, value_output


class TrainerNetwork:
    def __init__(self, encoder):
        self.encoder = encoder
        self.num_classes = self.encoder.num_points()
        self.board_input = Input(shape=encoder.shape_for_keras(), name='board_input')  # (None, 19, 19, 11)
        self.name = 'trainer3b'
        self.output = self.define_layers(0.01)

    def define_layers(self, reg_lambda):

        # trainer3b: 100/3:     0.0052 -> 0.0028 -> 0.0026 (BEZNADZIEJA!)
        # trainer3b: 1000/3:    0.0060 -> 0.0073 -> 0.0087
        # trainer3b: 5000/5:    0.01 -> ... -> 0.0363
        net = ZeroPadding2D(padding=3)(self.board_input)
        net = Conv2D(48, (7, 7), activation='relu', kernel_regularizer=l2(reg_lambda))(net)

        net = ZeroPadding2D(padding=2)(net)
        net = Conv2D(32, (5, 5), activation='relu', kernel_regularizer=l2(reg_lambda))(net)

        net = ZeroPadding2D(padding=2)(net)
        net = Conv2D(32, (5, 5), activation='relu', kernel_regularizer=l2(reg_lambda))(net)

        net = ZeroPadding2D(padding=2)(net)
        net = Conv2D(32, (5, 5), activation='relu', kernel_regularizer=l2(reg_lambda))(net)

        flat = Flatten()(net)
        output = Dense(self.num_classes, activation='softmax', kernel_regularizer=l2(reg_lambda))(flat)

        return output

        # trainer5: 3 x 0.0023
        # net = ZeroPadding2D(padding=3)(self.board_input)
        # net = Conv2D(48, (7, 7))(net)
        # net = BatchNormalization()(net)
        # net = LeakyReLU(alpha=0.1)(net)
        #
        # net = ZeroPadding2D(padding=2)(net)
        # net = Conv2D(32, (5, 5))(net)
        # net = BatchNormalization()(net)
        # net = LeakyReLU(alpha=0.1)(net)
        #
        # # Skip connection from layer 2 to layer 4
        # skip = net
        #
        # net = ZeroPadding2D(padding=2)(net)
        # net = Conv2D(32, (5, 5))(net)
        # net = BatchNormalization()(net)
        # net = LeakyReLU(alpha=0.1)(net)
        #
        # net = ZeroPadding2D(padding=2)(net)
        # net = Conv2D(32, (5, 5))(net)
        # net = BatchNormalization()(net)
        #
        # # Add skip connection to layer 4
        # net = Add()([net, skip])
        #
        # net = LeakyReLU(alpha=0.1)(net)
        #
        # flat = Flatten()(net)
        # flat = Dropout(0.5)(flat)
        # output = Dense(self.num_classes, activation='softmax')(flat)
        #
        # return output


        # trainer4 = 0.0042 -> 0.0047 -> 0.0042
        # net = ZeroPadding2D(padding=3)(self.board_input)
        # net = Conv2D(48, (7, 7))(net)
        # net = BatchNormalization()(net)
        # net = LeakyReLU(alpha=0.1)(net)
        #
        # net = ZeroPadding2D(padding=2)(net)
        # net = Conv2D(32, (5, 5))(net)
        # net = BatchNormalization()(net)
        # net = LeakyReLU(alpha=0.1)(net)
        #
        # net = ZeroPadding2D(padding=2)(net)
        # net = Conv2D(32, (5, 5))(net)
        # net = BatchNormalization()(net)
        # net = LeakyReLU(alpha=0.1)(net)
        #
        # net = ZeroPadding2D(padding=2)(net)
        # net = Conv2D(32, (5, 5))(net)
        # net = BatchNormalization()(net)
        # net = LeakyReLU(alpha=0.1)(net)
        #
        # flat = Flatten()(net)
        # flat = Dropout(0.5)(flat)
        # output = Dense(self.num_classes, activation='softmax')(flat)
        #
        # return output


        # trainer3a: 0.0035 -> 0.0045 -> 0.0025 (BEZNADZIEJA!)
        # conv_regularizer = l2(0.01)
        #
        # net = ZeroPadding2D(padding=3)(self.board_input)
        # net = Conv2D(48, (7, 7), kernel_regularizer=conv_regularizer)(net)
        # net = BatchNormalization()(net)
        # net = LeakyReLU(alpha=0.1)(net)
        #
        # net = ZeroPadding2D(padding=2)(net)
        # net = Conv2D(32, (5, 5), kernel_regularizer=conv_regularizer)(net)
        # net = BatchNormalization()(net)
        # net = LeakyReLU(alpha=0.1)(net)
        #
        # skip = net
        #
        # net = ZeroPadding2D(padding=2)(net)
        # net = Conv2D(32, (5, 5), kernel_regularizer=conv_regularizer)(net)
        # net = BatchNormalization()(net)
        # net = LeakyReLU(alpha=0.1)(net)
        #
        # net = ZeroPadding2D(padding=2)(net)
        # net = Conv2D(32, (5, 5), kernel_regularizer=conv_regularizer)(net)
        # net = BatchNormalization()(net)
        #
        # net = Add()([net, skip])
        #
        # net = LeakyReLU(alpha=0.1)(net)
        #
        # flat = Flatten()(net)
        # flat = Dropout(0.5)(flat)
        # output = Dense(self.num_classes, activation='softmax')(flat)
        #
        # return output



# class TrainerNetwork:
#     def __init__(self, encoder):
#         self.encoder = encoder
#         self.num_classes = 361
#         self.board_input = Input(shape=(19, 19, 11), name='board_input')  # (None, 19, 19, 11)
#         self.name = 'trainer2'
#         self.output = self.define_layers()

        # trainer: steady but slow progress / tried on laptops
        # net = ZeroPadding2D(padding=3)(self.board_input)
        # net = Conv2D(48, (7, 7), activation='relu')(net)
        #
        # net = ZeroPadding2D(padding=2)(net)
        # net = Conv2D(32, (5, 5), activation='relu')(net)
        #
        # net = ZeroPadding2D(padding=2)(net)
        # net = Conv2D(32, (5, 5), activation='relu')(net)
        #
        # net = ZeroPadding2D(padding=2)(net)
        # net = Conv2D(32, (5, 5), activation='relu')(net)
        #
        # flat = Flatten()(net)
        # output = Dense(self.num_classes, activation='softmax')(flat)
        #
        # return output



