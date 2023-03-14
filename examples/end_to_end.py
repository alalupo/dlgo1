import h5py
import os
from pathlib import Path
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent
from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.simple import SimpleEncoder
from dlgo.httpfrontend import get_web_app
from dlgo.networks import small


def main():
    go_board_rows, go_board_cols = 19, 19
    nb_classes = go_board_rows * go_board_cols
    encoder = SimpleEncoder((go_board_rows, go_board_cols))
    processor = GoDataProcessor(encoder=encoder.name())

    generator = processor.load_go_data('train', 100, use_generator=True)
    batch_size = 128
    num_samples = generator.get_num_samples(batch_size=batch_size, num_classes=nb_classes)
    print(f'The number of samples: {num_samples}')
    X, y = [], []
    data_gen = generator.generate(batch_size=batch_size, num_classes=nb_classes)
    for i in range(num_samples // batch_size):
        X_batch, y_batch = next(data_gen)
        X.append(X_batch)
        y.append(y_batch)
    X = np.concatenate(X)
    y = np.concatenate(y)

    # X, y = processor.load_go_data(num_samples=100, use_generator=True)

    input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
    model = Sequential()
    network_layers = small.layers(input_shape)
    for layer in network_layers:
        model.add(layer)
    model.add(Dense(nb_classes, activation='softmax', name='output_layer'))
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    model.fit(X, y, batch_size=128, epochs=1, verbose=1)

    deep_learning_bot = DeepLearningAgent(model, encoder)

    path = Path(__file__)
    project_lvl_path = path.parents[1]
    data_directory_name = 'agents'
    data_directory = project_lvl_path.joinpath(data_directory_name)
    filename = 'deep_bot.h5'
    file_path = str(data_directory.joinpath(filename))

    deep_learning_bot.serialize(h5py.File(file_path, "w"))

    model_file = h5py.File(file_path, "r")
    bot_from_file = load_prediction_agent(model_file)

    web_app = get_web_app({'predict': bot_from_file})
    web_app.run()


if __name__ == '__main__':
    main()
