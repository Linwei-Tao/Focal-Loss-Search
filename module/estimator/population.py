import logging
import random

import torch

from utils import gumbel_softmax_v1 as gumbel_softmax


class Population(object):

    def __init__(self, batch_size, tau, is_gae=False):
        # TODO: is_gae
        super(Population, self).__init__()
        self._individual = []
        self._fitness = []
        self.batch_size = batch_size
        self.tau = tau

    def select(self, which):
        if 'highest' == which:
            index = self._fitness.index(max(self._fitness))
        elif 'lowest' == which:
            index = self._fitness.index(min(self._fitness))
        elif 'random' == which:
            index = random.randint(0, len(self._fitness)-1)
        else:
            raise ValueError('unknown argument `which`: %s' % which)
        return index, self._individual[index], self._fitness[index]

    def remove(self, which):
        if 'highest' == which:
            index = self._fitness.index(max(self._fitness))
        elif 'lowest' == which:
            index = self._fitness.index(min(self._fitness))
        elif 'random' == which:
            index = random.randint(0, len(self._fitness)-1)
        else:
            raise ValueError('unknown argument `which`: %s' % which)
        del self._individual[index]
        del self._fitness[index]
        return index

    def get_batch(self, batch_size=None, tau=None):
        if batch_size is None: batch_size = self.batch_size
        if tau is None: tau = self.tau

        length = len(self)
        if batch_size > length:
            logging.warning('required batch_size (%d) is larger than memory size (%d)', batch_size, length)

        indices = [i for i in range(length)]
        random.shuffle(indices)

        individual = []
        fitness = []
        batch = []
        for idx in indices:
            (a_normal, g_normal), (a_reduce, g_reduce) = self._individual[idx]
            weights = torch.cat([gumbel_softmax(a_normal, tau=tau, dim=-1, g=g_normal),
                                 gumbel_softmax(a_reduce, tau=tau, dim=-1, g=g_reduce)])
            individual.append(weights)
            fitness.append(self._fitness[idx])
            if len(individual) >= batch_size:
                batch.append((torch.stack(individual), torch.stack(fitness)))
                individual = []
                fitness = []
        if len(individual) > 0:
            batch.append((torch.stack(individual), torch.stack(fitness)))
        return batch

    def append(self, individual, fitness):
        self._individual.append(individual)
        self._fitness.append(fitness)

    def state_dict(self):
        return {'_individual': self._individual,
                '_fitness': self._fitness}

    def load_state_dict(self, state_dict):
        self._individual = state_dict['_individual']
        self._fitness = state_dict['_fitness']

    def __len__(self):
        len_i = len(self._individual)
        len_f = len(self._fitness)
        assert len_i == len_f
        return len_i
