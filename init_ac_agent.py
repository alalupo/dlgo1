import argparse
import os

import h5py
import keras.backend as K
from keras.models import Model, save_model

from dlgo.encoders.base import get_encoder_by_name
from dlgo.networks.func_networks import AgentCriticNetwork


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    K.clear_session()

    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', '-bs', type=int, default=19)
    parser.add_argument('--output_file', '-of')
    args = parser.parse_args()

    encoder = get_encoder_by_name('simple', args.board_size)
    network = AgentCriticNetwork(encoder=encoder)
    model = Model(inputs=network.board_input,
                  outputs=[network.policy_output, network.value_output])

    with h5py.File(args.output_file, 'w') as outf:
        save_model(filepath=outf, model=model)


if __name__ == '__main__':
    main()
