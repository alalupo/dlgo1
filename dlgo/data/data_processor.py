import gzip
import logging
import multiprocessing
import os
import os.path
import shutil
import sys
import tarfile
from multiprocessing import get_context

import numpy as np

from dlgo.data.generator import DataGenerator
from dlgo.data.index_processor import KGSIndex
from dlgo.data.sampling import Sampler
from dlgo.encoders.base import get_encoder_by_name
from dlgo.goboard_fast import Board, GameState, Move
from dlgo.gosgf import Sgf_game
from dlgo.gotypes import Player, Point
from dlgo.tools.file_finder import FileFinder

logger = logging.getLogger('trainingLogger')


def worker(jobinfo):
    try:
        clazz, encoder, zip_file, data_file_name, game_list, board_size, data_type, num_samples = jobinfo
        clazz(encoder=encoder, board_size=board_size).read_zip(
            zip_file, data_file_name, game_list, data_type, num_samples)
    except (KeyboardInterrupt, SystemExit):
        raise Exception('>>> Exiting child process.')


class GoDataProcessor:
    total_train_samples = 0
    processed_train_samples = 0
    total_test_samples = 0
    processed_test_samples = 0

    def __init__(self, encoder='simple', board_size=19):
        self.encoder_string = encoder
        self.board_size = board_size
        self.encoder = get_encoder_by_name(encoder, self.board_size)
        finder = FileFinder()
        self.data_dir = finder.data_dir
        self.test_ratio = 0.2

    def load_go_data(self, num_samples, data_type='train'):
        index = KGSIndex(data_directory=self.data_dir)
        index.download_files()
        sampler = Sampler(num_test_games=np.floor(num_samples * self.test_ratio))
        samples = sampler.draw_data(data_type, num_samples)
        logger.info(f'{len(samples)} SAMPLES:')
        logger.info(f'{samples}')
        self.map_to_workers(data_type, samples, num_samples)
        return DataGenerator(self.data_dir, samples, self.board_size, data_type)

    def map_to_workers(self, data_type, samples, num_samples):
        zip_names = set()
        indices_by_zip_name = {}
        for filename, index in samples:
            zip_names.add(filename)
            if filename not in indices_by_zip_name:
                indices_by_zip_name[filename] = []
            indices_by_zip_name[filename].append(index)

        zips_to_process = []
        for zip_name in zip_names:
            base_name = zip_name.replace('.tar.gz', '')
            data_file_name = base_name + data_type
            if not os.path.isfile(self.data_dir.joinpath(data_file_name)):
                zips_to_process.append((self.__class__, self.encoder_string, zip_name,
                                        data_file_name, indices_by_zip_name[zip_name], self.board_size,
                                        data_type, num_samples))

        cores = multiprocessing.cpu_count()  # Determine number of CPU cores and split work load among them
        pnum = cores  # By default, pnum = cores but can be set to 1 if no multiprocessing needed
        print(f'The number of CPU: {cores}')
        print(f'The actual number of parallel processes: {pnum}')
        with get_context("spawn").Pool(processes=pnum, initializer=self.start_process) as pool:
            p = pool.map_async(worker, zips_to_process)
            try:
                _ = p.get()
            except KeyboardInterrupt:  # Caught keyboard interrupt, terminating workers
                print(f'Keyboard interrupt')
                pool.terminate()
                pool.join()
                sys.exit(-1)

    def read_zip(self, zip_file_name, data_file_name, game_list, data_type, num_samples):
        tar_file = self.unzip_data(zip_file_name)
        zip_file = tarfile.open(self.data_dir.joinpath(tar_file))
        name_list = zip_file.getnames()
        for index in game_list:
            moves, sgf = self.unpack_game(zip_file, index, name_list)
            game_features, game_labels = self.encode_game(data_type, index, moves, sgf)
            self.handle_saving(game_features, game_labels, data_file_name, index)
            self.progress_bar(game_list.index(index), len(game_list), 'games.')
        self.progress_bar(len(game_list), len(game_list), 'games.')  # to show 100%
        self.show_total_progress(data_type, num_samples, game_list)

    def handle_saving(self, features, labels, data_file_name, index):
        feature_path = self.data_dir.joinpath(f'{data_file_name}_features_{index}')
        label_path = self.data_dir.joinpath(f'{data_file_name}_labels_{index}')
        np.save(feature_path, features)
        np.save(label_path, labels)

    def unpack_game(self, zip_file, index, name_list):
        moves = self.num_moves_in_index(zip_file, index, name_list)
        name = name_list[index + 1]
        if not name.endswith('.sgf'):
            logger.warning(f'NAME {name} IS NOT A VALID SGF')
            raise ValueError(name + ' is not a valid sgf')
        sgf_content = zip_file.extractfile(name).read()
        return moves, Sgf_game.from_string(sgf_content)

    def encode_game(self, data_type, index, moves, sgf):
        # shape = self.encoder.shape_for_others()
        shape = self.encoder.shape()
        feature_shape = tuple(np.insert(shape, 0, np.asarray([moves])))
        features = np.zeros(feature_shape)
        labels = np.zeros((moves,))
        game_state, first_move_done = self.get_handicap(sgf, self.board_size)
        counter = 0
        for item in sgf.main_sequence_iter():
            color, move_tuple = item.get_move()
            point = None
            if color is not None:
                if move_tuple is not None:
                    row, col = move_tuple
                    point = Point(row + 1, col + 1)
                    move = Move.play(point)
                else:
                    move = Move.pass_turn()
                if first_move_done and point is not None:
                    features[counter] = self.encoder.encode(game_state)
                    labels[counter] = self.encoder.encode_point(point)
                    counter += 1
                game_state = game_state.apply_move(move)
                first_move_done = True
        return features, labels

    def show_total_progress(self, data_type, num_samples, game_list):
        print(f'')
        if data_type == 'train':
            GoDataProcessor.total_train_samples = num_samples
            GoDataProcessor.processed_train_samples += len(game_list)
            print(
                f'TOTAL PROGRESS: {GoDataProcessor.processed_train_samples}/{GoDataProcessor.total_train_samples}')
        else:
            GoDataProcessor.total_test_samples = int(num_samples * self.test_ratio)
            GoDataProcessor.processed_test_samples += len(game_list)
            print(
                f'TOTAL PROGRESS: {GoDataProcessor.processed_test_samples}/{GoDataProcessor.total_test_samples}')

    def unzip_data(self, zip_file_name):
        this_gz = gzip.open(self.data_dir.joinpath(zip_file_name))
        tar_file = zip_file_name[0:-3]
        this_tar = open(self.data_dir.joinpath(tar_file), 'wb')
        shutil.copyfileobj(this_gz, this_tar)
        this_tar.close()
        return tar_file

    @staticmethod
    def get_handicap(sgf, board_size):  # Get handicap stones
        go_board = Board(board_size, board_size)
        first_move_done = False
        move = None
        game_state = GameState.new_game(board_size)
        if sgf.get_handicap() is not None and sgf.get_handicap() != 0:
            for setup in sgf.get_root().get_setup_stones():
                for move in setup:
                    row, col = move
                    go_board.place_stone(Player.black, Point(row + 1, col + 1))  # black gets handicap
            first_move_done = True
            game_state = GameState(go_board, Player.white, None, move)
        return game_state, first_move_done

    def num_total_moves(self, zip_file, game_list, name_list):
        total_moves = 0
        for index in game_list:
            name = name_list[index + 1]
            if name.endswith('.sgf'):
                sgf_content = zip_file.extractfile(name).read()
                sgf = Sgf_game.from_string(sgf_content)
                game_state, first_move_done = self.get_handicap(sgf, board_size=self.board_size)

                num_moves = 0
                for item in sgf.main_sequence_iter():
                    color, move = item.get_move()
                    if color is not None:
                        if first_move_done:
                            num_moves += 1
                        first_move_done = True
                total_moves = total_moves + num_moves
            else:
                raise ValueError(name + ' is not a valid sgf')
        return total_moves

    def num_moves_in_index(self, zip_file, index, name_list):
        moves = 0
        name = name_list[index + 1]
        if name.endswith('.sgf'):
            sgf_content = zip_file.extractfile(name).read()
            sgf = Sgf_game.from_string(sgf_content)
            game_state, first_move_done = self.get_handicap(sgf, board_size=self.board_size)

            num_moves = 0
            for item in sgf.main_sequence_iter():
                color, move = item.get_move()
                if color is not None:
                    if first_move_done:
                        num_moves += 1
                    first_move_done = True
            moves = moves + num_moves
        else:
            raise ValueError(name + ' is not a valid sgf')
        return moves

    @staticmethod
    def start_process():
        print(f'Starting {multiprocessing.current_process().name}')

    @staticmethod
    def progress_bar(count_value, total, suffix=''):
        bar_length = 100
        filled_up_Length = int(round(bar_length * count_value / float(total)))
        percentage = round(100.0 * count_value / float(total), 1)
        bar = '=' * filled_up_Length + '-' * (bar_length - filled_up_Length)
        sys.stdout.write('[%s] %s%s %s\r' % (bar, percentage, '%', suffix))
        sys.stdout.flush()
