import argparse
import logging.config
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import h5py
import tensorflow as tf

keras = tf.keras
from keras.optimizers import SGD, Adam
from keras.models import Model, save_model

from dlgo.tools.file_finder import FileFinder
from dlgo.encoders.base import get_encoder_by_name
from dlgo.networks.func_networks import AgentCriticNetwork


logger = logging.getLogger('acTrainingLogger')


def main():
    logger.info('INITIATOR: STARTED')

    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', '-size', type=int, default=19)
    args = parser.parse_args()

    board_size = args.board_size

    initiator = Initiator(board_size)
    initiator.create_model()

    logger.info('INITIATOR: FINISHED')


class Initiator:
    def __init__(self, board_size, learning_rate=0.007):
        self.board_size = board_size
        self.encoder = get_encoder_by_name('simple', self.board_size)
        self.network = AgentCriticNetwork(encoder=self.encoder)
        self.model_filename = f'model_bs{self.board_size}_{self.network.name}_v1.h5'
        self.model_out_path = self.get_model_path(self.model_filename)
        self.learning_rate = learning_rate
        logger.info(f'BOARD SIZE: {self.board_size}')
        logger.info(f'ENCODER: {self.encoder}')
        logger.info(f'NETWORK: {self.network.name}')
        logger.info(f'OUTPUT FILE: {self.model_out_path}')
        logger.info(f'LEARNING RATE: {self.learning_rate}')

    @staticmethod
    def get_model_path(model_out):
        finder = FileFinder()
        return finder.get_model_full_path(model_out)

    def create_model(self):
        model = Model(inputs=self.network.board_input,
                      outputs=[self.network.policy_output, self.network.value_output])

        opt = SGD(learning_rate=self.learning_rate, clipnorm=1)
        model.compile(
            optimizer=opt,
            loss=['categorical_crossentropy', 'huber_loss'],
            loss_weights=[1, 0.5])

        with h5py.File(self.model_out_path, 'w') as outf:
            save_model(filepath=outf, model=model, save_format='h5')


if __name__ == '__main__':
    main()
