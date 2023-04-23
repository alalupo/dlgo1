import argparse
import logging.config
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import h5py
import tensorflow as tf

keras = tf.keras
from keras.optimizers import SGD
from keras.models import Model, save_model

from dlgo.tools.file_finder import FileFinder
from dlgo.encoders.base import get_encoder_by_name
from dlgo.networks.func_networks import AgentCriticNetwork

# logging.config.fileConfig('log_confs/ac_train_logging.conf')
logger = logging.getLogger('acTrainingLogger')


def main():
    logger.info('INITIATOR: STARTED')

    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', '-size', type=int, default=19)
    parser.add_argument('--output-file', '-out')
    args = parser.parse_args()

    board_size = args.board_size
    model_out = args.output_file

    initiator = Initiator(board_size, model_out)
    initiator.create_model()

    logger.info('INITIATOR: FINISHED')


class Initiator:
    def __init__(self, board_size, model_out, learning_rate=0.0001):
        self.board_size = board_size
        self.encoder = get_encoder_by_name('simple', self.board_size)
        self.network = AgentCriticNetwork(encoder=self.encoder)
        self.model_out_path = self.get_model_path(model_out)
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
        opt = SGD(learning_rate=0.0001, clipvalue=0.2)
        model.compile(
            optimizer=opt,
            loss=['categorical_crossentropy', 'mse'],
            loss_weights=[1.0, 0.5])

        with h5py.File(self.model_out_path, 'w') as outf:
            save_model(filepath=outf, model=model, save_format='h5')


if __name__ == '__main__':
    main()
