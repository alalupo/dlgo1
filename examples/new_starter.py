import argparse
import os
import shutil
from pathlib import Path
import h5py
from keras.models import load_model

from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent
from dlgo.agent.termination import TerminationAgent, PassWhenOpponentPasses, ResignLargeMargin
from dlgo.encoders.simple import SimpleEncoder
from dlgo.gotypes import Player
from dlgo.httpfrontend import get_web_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', '-size', type=int, default=19, required=False)
    parser.add_argument('--model', '-m', required=True)
    args = parser.parse_args()
    board_size = args.board_size
    filename = args.model
    starter = Starter(filename, board_size)
    starter.start()
    # starter.extract()


class Starter:
    def __init__(self, model_filename, board_size):
        self.encoder = SimpleEncoder((board_size, board_size))
        self.model_dir = 'checkpoints'
        self.model_name = model_filename
        self.model_copy_name = 'copy_' + self.model_name
        self.model_path = self.get_model_path()

    def get_model_path(self):
        path = Path(__file__)
        project_lvl_path = path.parents[1]
        model_dir_full_path = project_lvl_path.joinpath(self.model_dir)
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
        return model

    def create_bot(self):
        model = self.get_model()
        deep_learning_bot = DeepLearningAgent(model, self.encoder)
        with h5py.File(self.model_path, "w") as model_file:
            deep_learning_bot.serialize(model_file)
        with h5py.File(self.model_path, "r") as model_file:
            bot_from_file = load_prediction_agent(model_file)
            return TerminationAgent(bot_from_file, strategy=PassWhenOpponentPasses())
            # return TerminationAgent(bot_from_file, strategy=ResignLargeMargin(own_color=Player.white, cut_off_move=None, margin=100))

    def start(self):
        bot = self.create_bot()
        web_app = get_web_app({'predict': bot})
        web_app.run()

    def extract(self):
        model = self.get_model()
        outputs = [layer.output for layer in model.layers]
        for output in outputs:
            print(output)
        # print(f'output: {feature_layer.weights}')


if __name__ == '__main__':
    main()
