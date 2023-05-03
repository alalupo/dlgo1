import argparse
import logging.config
import os
import sys
from pathlib import Path

import tensorflow as tf

keras = tf.keras
from keras.models import load_model, save_model

project_path = Path(__file__).resolve().parent.parent
os.chdir(project_path)
sys.path.append(str(project_path))
sys.path.append(str(Path.cwd() / 'dlgo'))

from dlgo.agent.pg import PolicyAgent
from dlgo.encoders.base import get_encoder_by_name
from dlgo.exp.exp_reader import ExpReader
import h5py

logger = logging.getLogger('trainingLogger')


def main():
    logger.info('RL TRAINER: STARTED')
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-games', '-n', default=1000, type=int, required=False)
    parser.add_argument('experience', nargs='+')

    args = parser.parse_args()
    num_games = args.num_games
    experience_files = args.experience

    trainer = RLTrainer(num_games, experience_files)
    trainer.train()
    logger.info('RL TRAINER: FINISHED')


class RLTrainer:
    def __init__(self, num_games, exp_files):
        self.num_games = num_games
        self.board_size = 19
        self.batch_size = 1024
        self.encoder = get_encoder_by_name('simple', self.board_size)
        self.model_dir = Path.cwd() / 'models'
        self.model_sl_name = 'model_sl_strong_improved_100_1_epoch1.h5'
        self.model_rl_name = 'model_rl_strong_improved_100_1_epoch1.h5'
        self.learning_rate = 0.007
        self.exp_files = exp_files
        self.exp_paths = []
        if isinstance(exp_files, (list, tuple)):
            for exp_file in exp_files:
                exp_path = str(Path.cwd() / 'exp' / exp_file)
                self.exp_paths.append(exp_path)
        else:
            exp_path = str(Path.cwd() / 'exp' / exp_files)
            self.exp_paths.append(exp_path)

    def train(self):
        print(f'')
        print(f'>>>LOADING RL AGENT...')
        rl_agent = self.create_bot(self.model_sl_name)

        for exp_filename in self.exp_paths:
            print(f'')
            print(f'>>>LOADING EXPERIENCE: {exp_filename}...')
            generator = ExpReader(exp_file=exp_filename,
                                  batch_size=self.batch_size,
                                  num_planes=self.encoder.num_planes,
                                  board_size=self.board_size,
                                  seed=1234)
            print(f'>>> TRAINING RL MODEL...')
            rl_agent.train(
                generator,
                lr=self.learning_rate,
                clipnorm=1,
                batch_size=self.batch_size)

        self.save_rl_model(rl_agent.model)

    def create_bot(self, model_name):
        path = str(self.model_dir / model_name)
        model = self.get_sl_model(path)
        print(f'>>> RL bot for {model_name} has been created.')
        return PolicyAgent(model, self.encoder)

    def get_sl_model(self, model_path):
        model_file = None
        try:
            model_file = open(model_path, 'r')
        finally:
            model_file.close()
        with h5py.File(model_path, "r") as model_file:
            model = load_model(model_file)
        return model

    def save_rl_model(self, model):
        path = str(self.model_dir / self.model_rl_name)
        with h5py.File(path, 'w') as model_outf:
            save_model(model=model, filepath=model_outf, save_format='h5')
        print(f'>>> RL model has been saved.')


if __name__ == '__main__':
    main()
