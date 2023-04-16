import unittest
from pathlib import Path
from dlgo.data.parallel_processor import GoDataProcessor


class LoadingTest(unittest.TestCase):
    def setUp(self):
        self.board_size = 19
        self.samples_file = Path('test_samples.py')
        if Path(self.samples_file).is_file():
            Path.unlink(self.samples_file)
        for path in Path("./data").glob("*.npy"):
            if Path(path).is_file():
                Path.unlink(path)

    def tearDown(self):
        # Clean up the test file
        if Path(self.samples_file).is_file():
            Path.unlink(self.samples_file)
        for path in Path("./data").glob("*.npy"):
            if Path(path).is_file():
                Path.unlink(path)

    def test_loading(self):
        processor = GoDataProcessor('simple', self.board_size)
        generator = processor.load_go_data(num_samples=100, data_type='train')
        samples = generator.count_files()
        batch_size = 128
        num_samples = generator.get_num_samples(batch_size=batch_size)
        data_gen = generator.generate(batch_size=batch_size)
        X = []
        for i in range(num_samples // batch_size):
            X, y = next(data_gen)
        self.assertEqual(100, samples)
        self.assertEqual(128, X.shape[0])
        self.assertEqual(106, num_samples // batch_size)
