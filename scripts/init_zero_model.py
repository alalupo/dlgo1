import argparse
import logging.config
import os
import sys
from pathlib import Path
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

this_directory = os.path.dirname(__file__)
project_directory = os.path.dirname(this_directory)
sys.path.append(project_directory)
sys.path.append(this_directory)

import h5py
import tensorflow as tf

keras = tf.keras
from keras.optimizers import SGD, Adam
from keras.models import Model, save_model

from config import dlgo_directory

sys.path.append(dlgo_directory)

from dlgo.zero.encoder import ZeroEncoder
from dlgo.networks.network_architectures import Network

logger = logging.getLogger('zeroTrainingLogger')


def check_and_modify_parameter(param):
    match = re.search(r'(model\d*_)', param)
    if match:
        if '_sp_' not in param:
            param = param.replace(match.group(), match.group() + 'sp_')
    else:
        param = f'model_sp_{param}'
    return param


def main():
    logger.info('INITIATOR: STARTED')

    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', '-size', type=int, default=19)
    parser.add_argument('--model', '-m', required=True)

    args = parser.parse_args()
    board_size = args.board_size
    model_name = args.model

    model_name = check_and_modify_parameter(model_name)

    model_path = str(Path(project_directory) / 'models' / model_name)

    initiator = Initiator(board_size, model_path)
    initiator.create_model()

    logger.info('INITIATOR: FINISHED')


class Initiator:
    def __init__(self, board_size, model_path, learning_rate=0.007):
        self.board_size = board_size
        self.encoder = ZeroEncoder(self.board_size)
        self.network = Network(self.board_size)
        self.model_out_path = model_path
        self.learning_rate = learning_rate
        logger.info(f'BOARD SIZE: {self.board_size}')
        logger.info(f'ENCODER: {self.encoder}')
        logger.info(f'NETWORK: {self.network.name}')
        logger.info(f'OUTPUT FILE: {self.model_out_path}')
        logger.info(f'LEARNING RATE: {self.learning_rate}')

    def create_model(self):
        model = Model(
            inputs=[self.network.board_input],
            outputs=[self.network.policy_output, self.network.value_output])

        opt = Adam(learning_rate=self.learning_rate, clipnorm=1)
        model.compile(
            optimizer=opt,
            loss=['categorical_crossentropy', 'huber_loss'],
            loss_weights=[1, 0.5])

        with h5py.File(self.model_out_path, 'w') as outf:
            save_model(filepath=outf, model=model, save_format='h5')


if __name__ == '__main__':
    main()
