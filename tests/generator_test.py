import unittest
from pathlib import Path
import numpy as np

from dlgo.data.generator import DataGenerator
from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.base import get_encoder_by_name
from dlgo.gotypes import Point
from dlgo.gosgf import Sgf_game
from dlgo.goboard_fast import Board, GameState, Move


class GeneratorTest(unittest.TestCase):

    def test_generator(self):
        path = Path.cwd()
        dir = path / 'data'
        samples = [('KGS-2008-19-14002-.tar.gz', 9791), ('KGS-2012-19-13665-.tar.gz', 4261)]
        generator = DataGenerator(dir, samples, 19, 'train')
        print(f'num samples = {generator.get_num_samples()}')

