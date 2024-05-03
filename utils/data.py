from abc import abstractmethod
from math import ceil

import numpy as np


class Dataset:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, i):
        pass


class DataLoader:
    def __init__(self, dataset, batch, shuffle=True, drop_last=True, collate_fn=None):
        self.dataset = dataset
        self.indices = np.arange(len(dataset))
        self.collate_fn = collate_fn if collate_fn else self._collate_fn
        self.batch = batch
        self.drop_last = drop_last
        self.cache = {}
        if shuffle:
            np.random.shuffle(self.indices)
        self._idx = 0
        self.last_flag = False

    def __iter__(self):
        self._idx = 0
        self.last_flag = False
        return self

    @staticmethod
    def _collate_fn(*args):
        return tuple(map(np.stack, args)) if len(args) > 1 else np.stack(args[0])

    def __len__(self):
        return len(self.dataset) // self.batch if self.drop_last else ceil(len(self.dataset) / self.batch)

    def __next__(self):
        self._idx += self.batch
        if self._idx > len(self.dataset) and not self.last_flag:
            self.last_flag = True
            if self.drop_last:
                raise StopIteration
        elif self.last_flag:
            raise StopIteration
        batch_data = []
        for i in range(self._idx - self.batch, min(len(self.dataset), self._idx)):
            index = self.indices[i]
            if index in self.cache:
                data = self.cache[index]
            else:
                data = self.dataset[index]
                self.cache[index] = data
            batch_data.append(data)
        return self.collate_fn(*zip(*batch_data))


# 11 // 3 == 3 -> 4
# 12 // 3 == 4

if __name__ == "__main__":
    print(ceil(1.1))
