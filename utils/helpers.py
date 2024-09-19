import torch
import copy
from collections import deque


def get_params(model):
    params = {}
    for name, param in model.named_parameters():
        params[name] = nn.Parameter(param)
    return copy.deepcopy(params)
        

class LimitedQueue:
    def __init__(self, max_size=4):
        self.queue = deque(maxlen=max_size)

    def add(self, item):
        self.queue.append(item)

    def get_items(self):
        return list(self.queue)
