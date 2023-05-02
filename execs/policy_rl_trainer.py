import argparse
import logging.config

import tensorflow as tf

keras = tf.keras
from keras.models import load_model, save_model

from dlgo.agent.pg import PolicyAgent
from dlgo.encoders.base import get_encoder_by_name
from exp.simulate import Simulator
from dlgo.exp.exp_reader import ExpGenerator
import h5py

logger = logging.getLogger('trainingLogger')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-games', '-n', default=1000, type=int, required=False)
    # parser.add_argument('experience', nargs='+')

    args = parser.parse_args()
    num_games = args.num_games
    # experience_files = args.experience

    trainer = RLTrainer(num_games)
    trainer.train()


class RLTrainer:
    def __init__(self, num_games):
        self.num_games = num_games
        self.board_size = 19
        self.batch_size = 1024
        self.encoder = get_encoder_by_name('simple', self.board_size)
        self.model_sl_path = 'model_sl_policy.h5'
        self.model_rl_path = 'model_rl_policy.h5'
        self.learning_rate = 0.007
        self.exp_paths = []

    def train(self):
        print(f'')
        print(f'>>>LOADING AGENT')

        rl_agent = self.create_bot(self.model_sl_path)
        opponent = self.create_bot(self.model_sl_path)

        for i in range(0, self.num_games, 5000):
            games = min(self.num_games - i, 5000)
            simulator = Simulator(self.board_size, self.encoder.num_planes, games)
            path = simulator.experience_simulation(rl_agent, opponent)
            self.exp_paths.append(path)

        for exp_filename in self.exp_paths:
            print(f'')
            print(f'>>>LOADING EXPERIENCE: {exp_filename}')
            generator = ExpGenerator(exp_file=exp_filename,
                                     batch_size=self.batch_size,
                                     num_planes=self.encoder.num_planes,
                                     board_size=self.board_size,
                                     seed=1234)
            print(f'>>>MODEL TRAINING')
            rl_agent.train(
                generator,
                lr=self.learning_rate,
                clipnorm=1,
                batch_size=self.batch_size)

        print(f'>>> RL model is getting saved.')
        with h5py.File('model_rl_policy.h5', 'w') as model_outf:
            save_model(model=rl_agent.model, filepath=model_outf, save_format='h5')

    def create_bot(self, model_path):
        print(f'>>>Creating bot {model_path}...')
        model = self.get_model(model_path)
        return PolicyAgent(model, self.encoder)

    def get_model(self, model_path):
        model_file = None
        try:
            model_file = open(model_path, 'r')
        finally:
            model_file.close()
        with h5py.File(model_path, "r") as model_file:
            model = load_model(model_file)
        return model


if __name__ == '__main__':
    main()
