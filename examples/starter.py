import h5py
import os
from pathlib import Path

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model, save_model
import tensorflow as tf

from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent
from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.simple import SimpleEncoder
from dlgo.httpfrontend import get_web_app
from dlgo.networks import small
from dlgo import kerasutil


def main():
    go_board_rows, go_board_cols = 19, 19
    encoder = SimpleEncoder((go_board_rows, go_board_cols))

    path = Path(__file__)
    project_lvl_path = path.parents[1]
    data_directory_name = 'checkpoints'
    data_directory = project_lvl_path.joinpath(data_directory_name)
    filename = 'simple_large_model_epoch_7.h5'
    file_path = str(data_directory.joinpath(filename))
    if os.path.exists(file_path):
        print(f"{file_path} exists.")
    else:
        raise FileNotFoundError

    h5file = h5py.File(file_path)
    if h5file.__bool__():
        h5file.close()
    model_file = h5py.File(file_path, "r")

    model = load_model(model_file)
    print(model.summary())
    if model_file.__bool__():
        model_file.close()
    deep_learning_bot = DeepLearningAgent(model, encoder)

    model_file = h5py.File(file_path, "w")
    deep_learning_bot.serialize(model_file)

    if model_file.__bool__():
        model_file.close()
    model_file = h5py.File(file_path, "r")
    bot_from_file = load_prediction_agent(model_file)

    web_app = get_web_app({'predict': bot_from_file})
    web_app.run()


if __name__ == '__main__':
    main()
