import unittest
import numpy as np
from dlgo.data.mmap_np import NpArrayMapper


class MemmapTest(unittest.TestCase):
    def test_memmap_tuple_4_large(self):
        mapper = NpArrayMapper('tmp.npy', (65536, 19, 19, 11), np.float64)
        mapper.create_map()
        mapper.write_to_map(0, 1)
        mapper.write_to_map(1024, 1)
        value1 = mapper.read_from_map(0)
        tab = np.ones((1, 19, 19, 11))
        self.assertTrue((tab == value1).all())
        chunk = mapper.get_chunk(0)
        self.assertTrue((tab == chunk[0]).all())
        value2 = mapper.read_from_map(1024)
        self.assertTrue((tab == value2).all())

    def test_memmap_tuple_4_large_not_2_power(self):
        mapper = NpArrayMapper('tmp.npy', (65000, 19, 19, 11), np.float64)
        mapper.create_map()
        mapper.write_to_map(0, 1)
        mapper.write_to_map(1024, 1)
        value1 = mapper.read_from_map(0)
        tab = np.ones((1, 19, 19, 11))
        self.assertTrue((tab == value1).all())
        chunk = mapper.get_chunk(0)
        self.assertTrue((tab == chunk[0]).all())
        value2 = mapper.read_from_map(1024)
        self.assertTrue((tab == value2).all())

    def test_memmap_tuple_4_small(self):
        mapper = NpArrayMapper('tmp.npy', (512, 19, 19, 11), np.float64)
        mapper.create_map()
        mapper.write_to_map(0, 1)
        value1 = mapper.read_from_map(0)
        tab = np.ones((1, 19, 19, 11))
        self.assertTrue((tab == value1).all())
        chunk = mapper.get_chunk(0)
        self.assertTrue((tab == chunk[0]).all())

    def test_memmap_tuple_1_small(self):
        mapper = NpArrayMapper('tmp.npy', (512,), np.float64)
        mapper.create_map()
        mapper.write_to_map(0, 1)
        value1 = mapper.read_from_map(0)
        tab = np.ones((1,))
        self.assertTrue((tab == value1).all())
        chunk = mapper.get_chunk(0)
        self.assertTrue((tab == chunk[0]).all())

    def test_memmap_tuple_1_large_not_2_power(self):
        mapper = NpArrayMapper('tmp.npy', (65000,), np.float64)
        mapper.create_map()
        mapper.write_to_map(0, 1)
        value1 = mapper.read_from_map(0)
        tab = np.ones((1,))
        self.assertTrue((tab == value1).all())
        chunk = mapper.get_chunk(0)
        self.assertTrue((tab == chunk[0]).all())