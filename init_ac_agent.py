import argparse
import logging.config
import os
from pathlib import Path

import h5py
import keras.backend as K
import tensorflow as tf
from keras.models import Model

from dlgo import rl
from dlgo.encoders.base import get_encoder_by_name
from dlgo.networks import network_types
from dlgo.networks.ac_network import AgentCriticNetwork

logging.config.fileConfig('log_confs/train_logging.conf')
logger = logging.getLogger('trainingLogger')


def locate_directory():
    path = Path(__file__)
    project_lvl_path = path.parent
    model_directory_name = 'checkpoints'
    data_directory = project_lvl_path.joinpath(model_directory_name)
    if not os.path.exists(data_directory):
        raise FileNotFoundError
    return str(data_directory)


def show_intro():
    print(f'=== AGENT-CRITIC REINFORCEMENT LEARNING ===')
    print(f'*' * 80)
    print(f'Don\'t forget to clean up data directory of npy files before you start training a new model.')
    print(f'Put into the terminal the following command:')
    print(f'    find ./data/ -name \*.npy -delete')
    network_types.show_data_format()
    print(f'*' * 80)


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logger.info('Started')
    show_intro()
    K.clear_session()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    batch_size = 128

    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', '-bs', type=int, default=19)
    parser.add_argument('--output_file', '-of')
    args = parser.parse_args()

    encoder = get_encoder_by_name('simple', args.board_size)
    network = AgentCriticNetwork(encoder=encoder)
    model = Model(inputs=network.board_input,
                  outputs=[network.policy_output, network.value_output])

    new_agent = rl.ACAgent(model, encoder)
    with h5py.File(args.output_file, 'w') as outf:
        new_agent.serialize(outf)

    logger.info('Finished')


if __name__ == '__main__':
    main()
