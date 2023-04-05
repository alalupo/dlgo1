from pathlib import Path
import os
import logging
import logging.config

from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt

from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.simple import SimpleEncoder
from dlgo.networks import network_types

logging.config.fileConfig('train_logging.conf')
logger = logging.getLogger('trainingLogger')
# log_handler1 = logging.handlers.console
# log_handler2 = logging.handlers.file
# logger.addHandler(log_handler1)
# logger.addHandler(log_handler2)

def locate_directory():
    path = Path(__file__)
    project_lvl_path = path.parent
    model_directory_name = 'checkpoints'
    data_directory = project_lvl_path.joinpath(model_directory_name)
    if os.path.exists(data_directory):
        print(f"{data_directory} exists.")
    else:
        raise FileNotFoundError
    return str(data_directory)


def show_intro():
    print(f'********************************************************************************************')
    print(f'Don\'t forget to clean up data directory of npy files before you start training a new model.')
    print(f'Put into the terminal the following command:')
    print(f'    find ./data/ -name \*.npy -delete')
    network_types.show_data_format()
    print(f'********************************************************************************************')


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # logging.basicConfig(filename='trainer.log', format='%(levelname)s: %(asctime)s %(message)s', level=logging.INFO)

    logger.info('Started')
    show_intro()
    rows, cols = 19, 19
    encoder = SimpleEncoder((rows, cols))
    input_shape = (rows, cols, encoder.num_planes)
    network = network_types.SmallNetwork(input_shape)
    num_games = 5000
    epochs = 3
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    batch_size = 128
    trainer = Trainer(network, encoder, optimizer, num_games, epochs, rows, cols)
    trainer.train_model(optimizer, batch_size)
    logger.info('Finished')


class Trainer:
    def __init__(self, network, encoder, optimizer, num_games=100, num_epochs=5, rows=19, cols=19):
        self.network = network
        self.encoder = encoder
        self.optimizer = optimizer
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
        logger.info(f'Model summary:')
        logger.info(model.summary())
        logger.info(f'********************************************************************************************')
        return model

    def train_model(self, optimizer='adadelta', batch_size=128):
        processor = GoDataProcessor(encoder=self.encoder.name())
        generator = processor.load_go_data('train', num_samples=self.num_games, use_generator=True)
        test_generator = processor.load_go_data('test', num_samples=self.num_games, use_generator=True)
        logger.info(f'>>>Model training: generators loaded')
        checkpoint_dir = locate_directory()
        encoder_name = self.encoder.name()
        network_name = self.network.name

        logger.debug(f'>>>Model training: compiling...')
        # self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.model.compile(optimizer=self.optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        logger.debug(f'>>>Model training: fitting...')
        history = self.model.fit(
            generator.generate(batch_size, self.num_classes),
            epochs=self.epochs,
            steps_per_epoch=generator.get_num_samples() / batch_size,
            validation_data=test_generator.generate(batch_size, self.num_classes),
            validation_steps=test_generator.get_num_samples() / batch_size,
            callbacks=[
                ModelCheckpoint(checkpoint_dir + '/model_' + encoder_name + '_' + network_name + '_epoch_{epoch}.h5',
                                save_weights_only=False,
                                save_best_only=True
                                )
            ])

        logger.debug(f'>>>Model training: evaluating...')
        score = self.model.evaluate(
            test_generator.generate(batch_size, self.num_classes),
            steps=test_generator.get_num_samples() / batch_size)

        logger.debug(f'Test loss: {score[0]}')
        logger.debug(f'Test accuracy: {score[1]}')
        logger.debug(f'Score total: {score}')

        # print(history.history)

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'bo', label='Train loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title(f'Train and validation loss (games: {self.num_games}, epochs: {self.epochs})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{checkpoint_dir}/graph_{encoder_name}_{network_name}_{self.num_games}_{self.epochs}_loss.png')

        plt.clf()

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        plt.plot(epochs, acc, 'bo', label='Train accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title(f'Train and validation accuracy (games: {self.num_games}, epochs: {self.epochs})')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'{checkpoint_dir}/graph_{encoder_name}_{network_name}_{self.num_games}_{self.epochs}_accuracy.png')

        # # serialize model to JSON
        # model_json = self.model.to_json()
        # with open(checkpoint_dir + '/' + encoder_name + '_' + network_name + '_' + 'model.json', "w") as json_file:
        #     json_file.write(model_json)
        # print("Saved model to disk")


if __name__ == '__main__':
    main()
