import torch
import copy
from collections import deque, OrderedDict
import torch.nn as nn
import pickle


def get_params(models, names, pretrained_params):

    params = OrderedDict()

    for model, name_model, pretrained_params in zip(models, names, pretrained_params):
        par = {}

        if name_model == 'Target_critic':
            params[name_model] = copy.deepcopy(params['Critic'])

        if pretrained_params is None:
            for name, param in model.named_parameters():
                if 'bias' in name:
                    init = torch.nn.init.constant_(param, 0.0)
                else:
                    init = torch.nn.init.xavier_normal_(param)
                par[name] = nn.Parameter(init)
        else:
            for name, param in model.named_parameters():
                par[name] = nn.Parameter(init)

        params[name_model] = copy.deepcopy(par)

    return params



def reset_params(params, keys, optimizers, lr):
    for key in keys:
        for name, param in params[key].items():
            if len(param.shape) == 1:
                init = torch.nn.init.constant_(param, 0.01)
            else:
                init = torch.nn.init.xavier_normal_(param)
            params[key][name] = nn.Parameter(init)

        optimizers[key] = Adam(params[key].values(), lr)

    return params, optimizers


class LimitedQueue:
    def __init__(self, max_size=4):
        self.queue = deque(maxlen=max_size)

    def add(self, item):
        self.queue.append(item)

    def get_items(self):
        return list(self.queue)


def get_transition_from_pi(conn, addr):
    data = []
    
    while True:
        packet = conn.recv(4096)
        if not packet: break
        data.append(packet)

    return pickle.loads(b"".join(data))
