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
    # nb_classes = go_board_rows * go_board_cols
    encoder = SimpleEncoder((go_board_rows, go_board_cols))
    # processor = GoDataProcessor(encoder=encoder.name())
    #
    # X, y = processor.load_go_data(num_samples=10)
    #
    # input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
    # model = Sequential()
    # network_layers = small.layers(input_shape)
    # for layer in network_layers:
    #     model.add(layer)
    # model.add(Dense(nb_classes, activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    #
    # model.fit(X, y, batch_size=128, epochs=1, verbose=1)

    path = Path(__file__)
    project_lvl_path = path.parents[1]
    data_directory_name = 'agents'
    data_directory = project_lvl_path.joinpath(data_directory_name)
    filename = 'deep_bot.h5'
    newname = 'deep_bot_new.h5'
    file_path = str(data_directory.joinpath(filename))
    if os.path.exists(file_path):
        print(f"{file_path} exists.")
    else:
        raise FileNotFoundError

    model = load_model(file_path)

    file_path = str(data_directory.joinpath(newname))
    if os.path.exists(file_path):
        print(f"{file_path} exists.")
    else:
        raise FileNotFoundError

    deep_learning_bot = DeepLearningAgent(model, encoder)
    model_file = h5py.File(file_path, "w")
    deep_learning_bot.serialize(model_file)

    model_file = h5py.File(file_path, "r")
    bot_from_file = load_prediction_agent(model_file)

    web_app = get_web_app({'predict': bot_from_file})
    web_app.run()


if __name__ == '__main__':
    main()
