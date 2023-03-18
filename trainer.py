from pathlib import Path
import os

from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.simple import SimpleEncoder
from dlgo.networks import network_types

from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint
import tensorflow as tf


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
    print(f'Don\'t forget to clean up data directory of npy files before you start training a new model.')
    print(f'Put into the terminal the following command:')
    print(f'find ./data/ -name \*.npy -delete')
    rows, cols = 19, 19
    encoder = SimpleEncoder((rows, cols))
    input_shape = (encoder.num_planes, rows, cols)
    network = network_types.MediumNetwork(input_shape)
    num_games = 250
    epochs = 3
    optimizer = 'adadelta'
    batch_size = 128
    trainer = Trainer(network, encoder, num_games, epochs, rows, cols)
    trainer.build_model()
    trainer.train_model(optimizer, batch_size)


class Trainer:
    def __init__(self, network, encoder, num_games=100, num_epochs=5, rows=19, cols=19):
        self.encoder = encoder
        self.network = network
        self.num_games = num_games
        self.epochs = num_epochs
        self.go_board_rows, self.go_board_cols = rows, cols
        self.num_classes = self.go_board_rows * self.go_board_cols
        self.model = self.build_model()

    def build_model(self):
        network_layers = self.network.layers()
        model = Sequential()
        for layer in network_layers:
            model.add(layer)
        model.add(Dense(self.num_classes, activation='softmax'))
        print(model.summary())
        return model

    def train_model(self, optimizer='adadelta', batch_size=128):
        processor = GoDataProcessor(encoder=self.encoder.name())
        generator = processor.load_go_data('train', num_samples=self.num_games, use_generator=True)
        test_generator = processor.load_go_data('test', num_samples=self.num_games, use_generator=True)
        checkpoint_dir = locate_directory()
        encoder_name = self.encoder.name()
        network_name = self.network.name

        # self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        self.model.fit(
            generator.generate(batch_size, self.num_classes),
            epochs=self.epochs,
            steps_per_epoch=generator.get_num_samples() / batch_size,
            validation_data=test_generator.generate(batch_size, self.num_classes),
            validation_steps=test_generator.get_num_samples() / batch_size,
            callbacks=[
                ModelCheckpoint(checkpoint_dir + '/' + encoder_name + '_' + network_name + '_model_epoch_{epoch}.h5',
                                save_weights_only=False,
                                save_best_only=True
                                )
            ])

        score = self.model.evaluate(
            test_generator.generate(batch_size, self.num_classes),
            steps=test_generator.get_num_samples() / batch_size)

        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        # # serialize model to JSON
        # model_json = self.model.to_json()
        # with open(checkpoint_dir + '/' + encoder_name + '_' + network_name + '_' + 'model.json', "w") as json_file:
        #     json_file.write(model_json)
        # print("Saved model to disk")


if __name__ == '__main__':
    main()
