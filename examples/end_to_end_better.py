import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(dir_path, '..'))
#
# import h5py
#
# from keras.models import Sequential
# from keras.layers import Dense
#
# from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent
# from dlgo.data.parallel_processor import GoDataProcessor
# from dlgo.encoders.simple import SimpleEncoder
# from dlgo.httpfrontend import get_web_app
# from dlgo.networks import large
#
#
# def main():
#     go_board_rows, go_board_cols = 19, 19
#     nb_classes = go_board_rows * go_board_cols
#     encoder = SimpleEncoder((go_board_rows, go_board_cols))
#     processor = GoDataProcessor(encoder=encoder.name())
#
#     generator = processor.load_go_data('train', 100, use_generator=True)
#     batch_size = 10
#     num_classes = 19 * 19
#     num_samples = generator.get_num_samples(batch_size=batch_size, num_classes=num_classes)
#     print(num_samples)
#     X, y = [], []
#     data_gen = generator.generate(batch_size=batch_size, num_classes=num_classes)
#     for i in range(num_samples // batch_size):
#         X, y = next(data_gen)
#
#     input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
#     model = Sequential()
#     network_layers = large.layers(input_shape)
#     for layer in network_layers:
#         model.add(layer)
#     model.add(Dense(nb_classes, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
#
#     model.fit(X, y, batch_size=128, epochs=20, verbose=1)
#
#     # dirname = '../checkpoints'
#     # filename = 'pumperla_small_model_epoch_5.h5'
#
#     dirname = '../agents'
#     filename = 'end_to_end_large_model_epoch_1.h5'
#     relative_path = f'{dirname}/{filename}'
#
#     deep_learning_bot = DeepLearningAgent(model, encoder)
#     deep_learning_bot.serialize(h5py.File(relative_path))
#
#     model_file = h5py.File(relative_path, "r")
#     bot_from_file = load_prediction_agent(model_file)
#
#     web_app = get_web_app({'predict': bot_from_file})
#     web_app.run()


import h5py

from keras.models import Sequential
from keras.layers import Dense

from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent
from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.httpfrontend import get_web_app
from dlgo.networks import large
# end::e2e_imports[]

# tag::e2e_processor[]
def main():
    go_board_rows, go_board_cols = 19, 19
    nb_classes = go_board_rows * go_board_cols
    encoder = SevenPlaneEncoder((go_board_rows, go_board_cols))
    processor = GoDataProcessor(encoder=encoder.name())

    X, y = processor.load_go_data(num_samples=100)
    # end::e2e_processor[]

    # tag::e2e_model[]
    input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
    model = Sequential()
    network_layers = large.layers(input_shape)
    for layer in network_layers:
        model.add(layer)
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    model.fit(X, y, batch_size=128, epochs=20, verbose=1)
    # end::e2e_model[]

    # tag::e2e_agent[]
    deep_learning_bot = DeepLearningAgent(model, encoder)
    deep_learning_bot.serialize(h5py.File("../agents/deep_bot_old.h5"))
    # end::e2e_agent[]

    # tag::e2e_load_agent[]
    model_file = h5py.File("../agents/deep_bot_old.h5", "r")
    bot_from_file = load_prediction_agent(model_file)

    web_app = get_web_app({'predict': bot_from_file})
    web_app.run()


if __name__ == '__main__':
    main()
