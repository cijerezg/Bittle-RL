import torch
import copy
from collections import deque, OrderedDict
import torch.nn as nn
import pickle
import numpy as np
import pdb


class hyper_params:
    def __init__(self, args):
        for key, value in args.items():
            setattr(self, key, value)


def get_params(models, names, pretrained_params):
    params = OrderedDict()

    for model, name_model, pretrained_params in zip(models, names, pretrained_params):
        par = {}

        if name_model == 'Target_critic':
            params[name_model] = copy.deepcopy(params['Critic'])

        if pretrained_params is None:
            for name, param in model.named_parameters():
                if len(param.shape) <= 1:
                    if 'bias' in name:
                        init = torch.nn.init.constant_(param, 0.0)
                    else:
                        init = torch.nn.init.constant_(param, 1.0)
                else:
                    init = torch.nn.init.xavier_normal_(param)
                par[name] = nn.Parameter(init)
        else:
            for name, param in model.named_parameters():
                par[name] = nn.Parameter(init)

        params[name_model] = copy.deepcopy(par)

    return params



class LimitedQueue:
    def __init__(self, shape, max_size=8):
        self.queue = deque(maxlen=max_size)
        for i in range(max_size):
            self.queue.append(np.zeros(shape, dtype=np.float32))            

    def add(self, item):
        self.queue.append(item)

    def get_items(self):
        return np.stack(self.queue)


def send_data(sender, data):
    for val in data:
        serial_val = pickle.dumps(data)
        sender.sendall(serial_val)
        
    
def get_data(conn):
    data = []
    
    while True:
        packet = conn.recv(4096)
        if not packet: break
        data.append(packet)

    return pickle.loads(b"".join(data))


class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError(f"Attribute {attr} not found")

    def __getstate__(self):
        return self

    def __setstate__(self, d):
        self = d
        
