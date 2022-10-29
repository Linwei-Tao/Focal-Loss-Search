import random
from collections import namedtuple, deque

import torch


Experience = namedtuple('Experience', ['weights', "nll", "acc", "ece"])


class Memory(object):
    """Memory"""

    def __init__(self, limit=128, batch_size=64):
        assert limit >= batch_size, 'limit (%d) should not less than batch size (%d)' % (limit, batch_size)
        super(Memory, self).__init__()
        self.limit = limit
        self.batch_size = batch_size
        self.memory = deque(maxlen=limit)

    def get_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        assert batch_size <= self.limit, \
            'require batch_size (%d) exceeds memory limit, should be less than %d' % (batch_size, self.limit)
        length = len(self)
        if batch_size > length:
            print('required batch_size ({}) is larger than memory size ({})'.format(batch_size, length))

        indices = [i for i in range(length)]
        random.shuffle(indices)
        weights = []
        nll = []
        acc = []
        ece = []
        batch = []
        for idx in indices:
            weights.append(self.memory[idx].weights)
            nll.append(self.memory[idx].nll)
            ece.append(self.memory[idx].ece)
            acc.append(self.memory[idx].acc)
            if len(nll) >= batch_size:
                batch.append((torch.stack(weights), torch.stack(nll), torch.stack(acc), torch.stack(ece)))
                weights = []
                nll = []
                acc = []
                ece = []
        if len(nll) > 0: batch.append((torch.stack(weights), torch.stack(nll), torch.stack(acc), torch.stack(ece)))
        return batch

    def append(self, weights, nll, acc, ece):
        self.memory.append(Experience(weights=weights, nll=nll, acc=acc, ece=ece))

    def state_dict(self):
        return {'limit': self.limit,
                'batch_size': self.batch_size,
                'memory': self.memory}

    def load_state_dict(self, state_dict):
        self.limit = state_dict['limit']
        self.batch_size = state_dict['batch_size']
        self.memory = state_dict['memory']

    def __len__(self):
        return len(self.memory)
