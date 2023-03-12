from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.oneplane import OnePlaneEncoder

from dlgo.networks import small
from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta


def main():
    go_board_rows, go_board_cols = 19, 19
    num_classes = go_board_rows * go_board_cols
    num_games = 100

    encoder = OnePlaneEncoder((go_board_rows, go_board_cols))
    processor = GoDataProcessor(encoder=encoder.name())
    generator = processor.load_go_data('train', num_games, use_generator=True)
    test_generator = processor.load_go_data('test', num_games, use_generator=True)

    input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
    network_layers = small.layers(input_shape)
    model = Sequential()
    for layer in network_layers:
        model.add(layer)
    model.add(Dense(num_classes, activation='softmax'))
    adadelta = Adadelta()
    model.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=['accuracy'])
    epochs = 150
    batch_size = 128
    model.fit(
        generator.generate(batch_size, num_classes),
        epochs=epochs,
        steps_per_epoch=generator.get_num_samples() / batch_size,
        validation_data=test_generator.generate(batch_size, num_classes),
        validation_steps=test_generator.get_num_samples() / batch_size,
        callbacks=[
            ModelCheckpoint('../checkpoints/small_model_epoch_{epoch}.h5')
        ])
    model.evaluate(
        test_generator.generate(batch_size, num_classes),
        steps=test_generator.get_num_samples() / batch_size)


if __name__ == '__main__':
    main()
