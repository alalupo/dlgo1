import numpy as np
import os
import tempfile
import h5py

from keras.models import load_model, save_model

from dlgo.agent.base import Agent
from dlgo.agent.helpers import is_point_an_eye
from dlgo import encoders
from dlgo.encoders import base
from dlgo import goboard_fast
from dlgo import kerasutil


class DeepLearningAgent(Agent):
    def __init__(self, model, encoder):
        Agent.__init__(self)
        self.model = model
        self.encoder = encoder

    def predict(self, game_state):
        encoded_state = self.encoder.encode(game_state)
        input_tensor = np.array([encoded_state])
        return self.model.predict(input_tensor)[0]

    def select_move(self, game_state):
        num_moves = self.encoder.board_width * self.encoder.board_height
        move_probs = self.predict(game_state)
        move_probs = move_probs ** 3
        eps = 1e-6
        move_probs = np.clip(move_probs, eps, 1 - eps)
        move_probs = move_probs / np.sum(move_probs)
        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(candidates, num_moves, replace=False, p=move_probs)
        for point_idx in ranked_moves:
            point = self.encoder.decode_point_index(point_idx)
            if game_state.is_valid_move(goboard_fast.Move.play(point)) and \
                    not is_point_an_eye(game_state.board, point, game_state.next_player):
                return goboard_fast.Move.play(point)
            return goboard_fast.Move.pass_turn()

    # def serialize(self, h5file):
    #     model_dir = os.path.join(os.path.dirname(h5file.filename), 'tmp-kerasmodel')
    #     os.makedirs(model_dir, exist_ok=True)
    #     h5file.create_group('encoder')
    #     h5file['encoder'].attrs['name'] = self.encoder.name()
    #     h5file['encoder'].attrs['board_width'] = self.encoder.board_width
    #     h5file['encoder'].attrs['board_height'] = self.encoder.board_height
    #     h5file.create_group('model')
    #     self.model.save(model_dir)

    def serialize(self, h5file):
        # h5file.create_group('encoder')
        # h5file['encoder'].attrs['name'] = self.encoder.name()
        # h5file['encoder'].attrs['board_width'] = self.encoder.board_width
        # h5file['encoder'].attrs['board_height'] = self.encoder.board_height
        # h5file.create_group('model')
        # with tempfile.TemporaryDirectory(dir='./') as tmpdir:
        #     tempfname = os.path.join(tmpdir, 'tmp-kerasmodel')
        #     save_model(self.model, tempfname)
        #     serialized_model = h5py.File(tempfname, 'r')
        #     root_item = serialized_model.get('/')
        #     serialized_model.copy(root_item, h5file['model'], 'kerasmodel')
        #     serialized_model.close()
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self.encoder.name()
        h5file['encoder'].attrs['board_width'] = self.encoder.board_width
        h5file['encoder'].attrs['board_height'] = self.encoder.board_height
        h5file.create_group('model')
        return h5file

    def save(self, h5file):
        os.makedirs(os.path.dirname("./tmp/"), exist_ok=True)
        with tempfile.TemporaryDirectory(dir='./tmp/') as tmpdir:
            tempfname = os.path.join(tmpdir, '/tmp-kerasmodel')
            save_model(self.model, tempfname)
            serialized_model = h5py.File(tempfname, 'r')
            root_item = serialized_model.get('/')
            serialized_model.copy(root_item, h5file['model'], 'kerasmodel')
            serialized_model.close()


    def load(self, h5file, custom_objects=None):
        # with tempfile.TemporaryDirectory(dir='./') as tmpdir:
        #     tempfname = os.path.join(tmpdir, 'tmp-kerasmodel')
        #     serialized_model = h5py.File(tempfname, 'w')
        #     root_item = h5file.get('model/kerasmodel')
        #     for attr_name, attr_value in root_item.attrs.items():
        #         serialized_model.attrs[attr_name] = attr_value
        #     for k in root_item.keys():
        #         h5file.copy(root_item.get(k), serialized_model, k)
        #     serialized_model.close()
        #     self.model = load_model(tempfname, custom_objects=custom_objects)
        #     encoder_name = h5file['encoder'].attrs['name']
        #     if not isinstance(encoder_name, str):
        #         encoder_name = encoder_name.decode('ascii')
        #     board_width = h5file['encoder'].attrs['board_width']
        #     board_height = h5file['encoder'].attrs['board_height']
        #     encoder = base.get_encoder_by_name(encoder_name, (board_width, board_height))
        #     # Create and return a new agent using the loaded model and encoder
        #     return DeepLearningAgent(self.model, encoder)

        with tempfile.TemporaryDirectory(dir='./tmp/') as tmpdir:
            tempfname = os.path.join(tmpdir, 'tmp-kerasmodel')
            serialized_model = h5py.File(tempfname, 'w')
            root_item = h5file.get('model/kerasmodel')
            for attr_name, attr_value in root_item.attrs.items():
                serialized_model.attrs[attr_name] = attr_value
            for k in root_item.keys():
                h5file.copy(root_item.get(k), serialized_model, k)
            serialized_model.close()
            self.model = load_model(tempfname, custom_objects=custom_objects)
            encoder_name = h5file['encoder'].attrs['name']
            if not isinstance(encoder_name, str):
                encoder_name = encoder_name.decode('ascii')
            board_width = h5file['encoder'].attrs['board_width']
            board_height = h5file['encoder'].attrs['board_height']
            encoder = base.get_encoder_by_name(encoder_name, (board_width, board_height))
            # Create and return a new agent using the loaded model and encoder
            return DeepLearningAgent(self.model, encoder)


def load_prediction_agent(h5file):
    model = kerasutil.load_model_from_hdf5_group(h5file['model'])
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']
    encoder = encoders.get_encoder_by_name(
        encoder_name, (board_width, board_height))
    return DeepLearningAgent(model, encoder)
