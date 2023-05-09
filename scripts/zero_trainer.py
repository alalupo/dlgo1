import argparse
import logging.config
import os
import sys
from pathlib import Path

import h5py
import tensorflow as tf

keras = tf.keras
from keras.models import save_model, load_model

this_directory = os.path.dirname(__file__)
project_directory = os.path.dirname(this_directory)
sys.path.append(project_directory)
sys.path.append(this_directory)

from dlgo.exp.zero_exp_reader import ZeroExpReader
from dlgo.zero.agent import ZeroAgent
from dlgo.zero.encoder import ZeroEncoder

# from dlgo.utils import print_board


logger = logging.getLogger('zeroTrainingLogger')


def main():
    logger.info('TRAINER: STARTED')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', required=True)
    parser.add_argument('experience', nargs='+')

    args = parser.parse_args()
    model_name = args.model
    experience_files = args.experience

    model_path = str(Path(project_directory) / 'models' / model_name)
    exp_paths = []
    if isinstance(experience_files, (list, tuple)):
        for exp_file in experience_files:
            exp_path = str(Path(project_directory) / 'exp' / exp_file)
            exp_paths.append(exp_path)
    else:
        exp_path = str(Path(project_directory) / 'exp' / experience_files)
        exp_paths.append(exp_path)

    board_size = 19

    trainer = ZeroTrainer(board_size, model_path, exp_paths)
    trainer.train()

    logger.info('TRAINER: FINISHED')


class ZeroTrainer:
    def __init__(self, board_size, model_sl_path: str, exp_paths: list[str]):
        self.board_size = board_size
        self.batch_size = 1024
        # Optimally, a few hundred rounds per move
        self.rounds_per_move = 50
        self.encoder = ZeroEncoder(self.board_size)
        self.model_sl_path = model_sl_path
        self.model_rl_path = Path(str(self.model_sl_path).replace('_sl_', '_rl_'))
        self.learning_rate = 0.007
        self.exp_paths = []
        if isinstance(exp_paths, (list, tuple)):
            for exp_file in exp_paths:
                self.exp_paths.append(exp_file)
        else:
            self.exp_paths.append(exp_paths)

    def train(self):
        print(f'')
        print(f'>>> LOADING AGENT...')
        agent = self.create_bot()

        for exp_filename in self.exp_paths:
            print(f'')
            print(f'>>> LOADING EXPERIENCE: {exp_filename}...')
            generator = ZeroExpReader(exp_file=exp_filename,
                                      batch_size=self.batch_size,
                                      seed=1234)
            print(f'>>> TRAINING MODEL...')
            agent.train(
                generator,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size)

        self.save_zero_model(agent.model)

    def create_bot(self):
        model = self.get_sl_model()
        print(f'>>> Bot for {self.model_sl_path} has been created.')
        return ZeroAgent(model, self.encoder)

    def get_sl_model(self):
        model_file = None
        try:
            model_file = open(self.model_sl_path, 'r')
        finally:
            model_file.close()
        with h5py.File(self.model_sl_path, "r") as model_file:
            model = load_model(model_file)
        return model

    def save_zero_model(self, model):
        with h5py.File(self.model_rl_path, 'w') as f:
            save_model(model=model, filepath=f, save_format='h5')
        print(f'>>> RL model has been saved at {self.model_rl_path}.')


if __name__ == '__main__':
    main()
