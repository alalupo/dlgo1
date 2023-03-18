import h5py
import os
import shutil
from pathlib import Path

from keras.models import load_model


from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent
from dlgo.agent.termination import TerminationAgent, PassWhenOpponentPasses
from dlgo.encoders.simple import SimpleEncoder
from dlgo.httpfrontend import get_web_app


def main():
    go_board_rows, go_board_cols = 19, 19
    encoder = SimpleEncoder((go_board_rows, go_board_cols))

    path = Path(__file__)
    project_lvl_path = path.parents[1]

    model_directory_name = 'checkpoints'
    data_directory = project_lvl_path.joinpath(model_directory_name)
    filename = 'simple_medium_model_epoch_2.h5'
    new_filename = 'new_' + filename
    model_path = str(data_directory.joinpath(filename))
    new_model_path = str(data_directory.joinpath(new_filename))
    if os.path.exists(model_path):
        print(f"{model_path} exists.")
    else:
        raise FileNotFoundError

    shutil.copy(model_path, new_model_path)
    model_path = new_model_path

    model_file = None
    try:
        model_file = open(model_path, 'r')
    finally:
        model_file.close()

    # load h5py and create model
    with h5py.File(model_path, "r") as model_file:
        print(f'Items: {model_file.items()}')
        print(f'Keys: {model_file.keys()}')
        model = load_model(model_file)
        config = model.get_config()
        print(f'Printing h5py config: {config}')
        print(model.summary())
        deep_learning_bot = DeepLearningAgent(model, encoder)

    with h5py.File(model_path, "w") as model_file:
        deep_learning_bot.serialize(model_file)

    with h5py.File(model_path, "r") as model_file:
        bot_from_file = load_prediction_agent(model_file)
        termination_bot = TerminationAgent(bot_from_file, strategy=PassWhenOpponentPasses())
        web_app = get_web_app({'predict': termination_bot})
        web_app.run()


if __name__ == '__main__':
    main()
