from pathlib import Path
import os

from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.simple import SimpleEncoder

from dlgo.networks import small, medium, large
from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint


def locate_directory():
    path = Path(__file__)
    project_lvl_path = path.parent
    data_directory_name = 'checkpoints'
    data_directory = project_lvl_path.joinpath(data_directory_name)
    if os.path.exists(data_directory):
        print(f"{data_directory} exists.")
    else:
        raise FileNotFoundError
    return str(data_directory)


def main():
    go_board_rows, go_board_cols = 19, 19
    num_classes = go_board_rows * go_board_cols
    num_games = 5000

    encoder = SimpleEncoder((go_board_rows, go_board_cols))
    processor = GoDataProcessor(encoder=encoder.name())
    generator = processor.load_go_data('train', num_games, use_generator=True)
    test_generator = processor.load_go_data('test', num_games, use_generator=True)

    input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
    network_layers = large.layers(input_shape)
    model = Sequential()
    for layer in network_layers:
        model.add(layer)
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    epochs = 7
    batch_size = 128
    checkpoint_dir = locate_directory()
    encoder_name = encoder.name()
    network_name = 'large'

    model.fit(
        generator.generate(batch_size, num_classes),
        epochs=epochs,
        steps_per_epoch=generator.get_num_samples() / batch_size,
        validation_data=test_generator.generate(batch_size, num_classes),
        validation_steps=test_generator.get_num_samples() / batch_size,
        callbacks=[
            ModelCheckpoint(checkpoint_dir + '/' + encoder_name + '_' + network_name + '_model_epoch_{epoch}.h5')
        ])
    model.evaluate(
        test_generator.generate(batch_size, num_classes),
        steps=test_generator.get_num_samples() / batch_size)


class Trainer:
    def __init__(self, network, encoder, num_games=100, num_epochs=5):
        self.encoder = encoder
        self.network = network
        self.num_games = num_games
        self.epochs = num_epochs
        self.go_board_rows, self.go_board_cols = 19, 19
        self.num_classes = self.go_board_rows * self.go_board_cols
        self.input_shape = (self.encoder.num_planes, self.go_board_rows, self.go_board_cols)
        self.model = self.build_model()

    def build_model(self):
        network_layers = self.network.layers()
        model = Sequential()
        for layer in network_layers:
            model.add(layer)
        model.add(Dense(self.num_classes, activation='softmax'))
        return model

    def train_model(self, optimizer='adadelta', batch_size=128):
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        processor = GoDataProcessor(encoder=self.encoder.name())
        generator = processor.load_go_data('train', num_samples=self.num_games, use_generator=True)
        test_generator = processor.load_go_data('test', num_samples=self.num_games, use_generator=True)
        checkpoint_dir = locate_directory()
        encoder_name = self.encoder.name()
        network_name = self.network.name

        self.model.fit(
            generator.generate(batch_size, self.num_classes),
            epochs=self.epochs,
            steps_per_epoch=generator.get_num_samples() / batch_size,
            validation_data=test_generator.generate(batch_size, self.num_classes),
            validation_steps=test_generator.get_num_samples() / batch_size,
            callbacks=[
                ModelCheckpoint(checkpoint_dir + '/' + encoder_name + '_' + network_name + '_model_epoch_{epoch}.h5')
            ])
        self.model.evaluate(
            test_generator.generate(batch_size, self.num_classes),
            steps=test_generator.get_num_samples() / batch_size)


if __name__ == '__main__':
    main()
