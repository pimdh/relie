import math
import numpy as np


class TensorLoader:
    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.i = 0
        self.idxs = None

    def __iter__(self):
        self.i = 0
        self.idxs = self.indices()
        return self

    def indices(self):
        idxs = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(idxs)
        return idxs

    def __next__(self):
        batch = self.dataset[self.i:self.i+self.batch_size]
        if len(batch[0]) == 0:
            raise StopIteration()
        self.i += self.batch_size
        return batch

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)


def cycle(it):
    """Infinite iterator cycle, re-inits iterator every time."""
    while True:
        for x in it:
            yield x
