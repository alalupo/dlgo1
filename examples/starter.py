import argparse
import os
import shutil

import h5py
import tensorflow as tf
from keras.models import load_model

from dlgo.agent.predict import DeepLearningAgent
from dlgo.agent.termination import TerminationAgent, PassWhenOpponentPasses
from dlgo.encoders.simple import SimpleEncoder
from dlgo.httpfrontend import get_web_app
from dlgo.tools.file_finder import FileFinder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', '-size', type=int, default=19, required=False)
    parser.add_argument('--model', '-m', required=True)
    args = parser.parse_args()
    board_size = args.board_size
    filename = args.model
    starter = Starter(filename, board_size)
    starter.start()


class Starter:
    def __init__(self, model_filename, board_size):
        self.encoder = SimpleEncoder((board_size, board_size))
        finder = FileFinder()
        self.model_dir = finder.model_dir
        self.model_name = model_filename
        self.model_copy_name = 'play_' + self.model_name
        self.model_path = self.get_model_path()

    def get_model_path(self):
        model_dir_full_path = self.model_dir
        model_path = str(model_dir_full_path.joinpath(self.model_name))
        model_copy_path = str(model_dir_full_path.joinpath(self.model_copy_name))
        if not os.path.exists(model_path):
            raise FileNotFoundError
        shutil.copy(model_path, model_copy_path)
        model_path = model_copy_path
        return model_path

    def get_model(self):
        model_file = None
        try:
            model_file = open(self.model_path, 'r')
        finally:
            model_file.close()
        with h5py.File(self.model_path, "r") as model_file:
            model = load_model(model_file)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        return model

    def create_bot(self):
        model = self.get_model()
        deep_learning_bot = DeepLearningAgent(model, self.encoder)
        return TerminationAgent(deep_learning_bot, strategy=PassWhenOpponentPasses())

    def start(self):
        bot = self.create_bot()
        web_app = get_web_app({'predict': bot})
        web_app.run()


if __name__ == '__main__':
    main()
