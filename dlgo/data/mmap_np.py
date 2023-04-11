import numpy as np
import os


class NpArrayMapper:
    def __init__(self, file_path, shape, dtype):
        self.file_path = file_path
        self.shape = shape
        self.dtype = dtype
        self.chunk_size = 1024 if 1024 <= self.shape[0] else self.shape[0]
        print(f'NpArrayMapper created with shape {self.shape}')

    def create_map(self):
        expected_size = self.shape[0]
        if len(self.shape) > 1:
            chunked_shape = (self.chunk_size,) + self.shape[1:]
        else:
            chunked_shape = (self.chunk_size,)
        print(f'EXPECTED SIZE: {expected_size}')
        # create the memory-mapped array on disk
        mmapped_array = np.memmap(self.file_path, dtype=self.dtype, shape=self.shape, mode='w+')
        # fill the memmap array with zeros (to initialize it)
        for i in range(0, self.shape[0], self.chunk_size):
            if (i + self.chunk_size) > self.shape[0]:
                chunk_0 = self.shape[0] - i
                if len(self.shape) > 1:
                    chunked_shape = (chunk_0,) + self.shape[1:]
                else:
                    chunked_shape = (chunk_0,)
            chunk = np.zeros(chunked_shape)
            if (i + self.chunk_size) > self.shape[0]:
                if len(self.shape) > 1:
                    # TODO: ValueError could not broadcast input shape from 1024 into 829
                    mmapped_array[i:i + chunk_0, :] = chunk
                else:
                    mmapped_array[i:i + chunk_0] = chunk
            else:
                if len(self.shape) > 1:
                    # TODO: ValueError could not broadcast input shape from 1024 into 829
                    mmapped_array[i:i + self.chunk_size, :] = chunk
                else:
                    mmapped_array[i:i + self.chunk_size] = chunk
        # get rid of the memory object now when the data is saved on the hard disk
        del mmapped_array

    def num_chunks(self):
        return self.shape[0] // self.chunk_size

    def get_size(self):
        my_array = np.memmap(self.file_path, dtype=self.dtype, shape=self.shape, mode='r')
        size = round(my_array.nbytes / 1000000, 2)
        del my_array
        return size

    def write_to_map(self, index, value):
        my_array = np.memmap(self.file_path, dtype=self.dtype, shape=self.shape, mode='r+')
        # calculate the starting index and offset of the chunk to write
        start_index = index // self.chunk_size * self.chunk_size
        offset = index % self.chunk_size
        # read in the chunk of the array containing the index to write
        chunk = my_array[start_index:start_index + self.chunk_size]
        # write the value to the chunk at the correct offset
        chunk[offset] = value
        # save the chunk back to disk
        my_array[start_index:start_index + self.chunk_size] = chunk

    def read_from_map(self, index):
        my_array = np.memmap(self.file_path, dtype=self.dtype, shape=self.shape, mode='r')
        # read a single value from the memmap array
        value = my_array[index]
        del my_array
        return value

    def get_chunk(self, n):
        my_array = np.memmap(self.file_path, dtype=self.dtype, shape=self.shape, mode='r')
        chunk = my_array[n * self.chunk_size:(n + 1) * self.chunk_size]
        del my_array
        return chunk

    # define a generator function to yield chunks of the array
    def chunk_generator(self, array):
        for i in range(0, array.shape[0], self.chunk_size):
            yield array[i:i + self.chunk_size]

    def clean_up(self):
        os.remove(str(self.file_path))
