import torch
import copy
from collections import deque, OrderedDict
import torch.nn as nn
import pickle
import numpy as np
import pdb
import datetime
import os
from pathlib import Path


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


def save_experiences(path, transition, step):
    np.savez(f'{path}/transition_{step}.npz', *transition)
       

def load_experiences(path, delete=True):
    if os.listdir(path):
        exps = []
        files = [os.path.join(path, file) for file in os.listdir(path)]
        files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        for exp_file in files:
            try:
                data = np.load(exp_file)
                dt_string = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
                if 'init' not in path:
                    joints = data['arr_0']
                    dist = data['arr_1']
                    a = data['arr_2']
                    np.savez(f'experiences_library/a_{dt_string}.npz', joints, dist, a)
                exps.append(data)
            except EOFError:
                print(f'unable to read experience with name {exp_file}')
            if delete:
                Path(exp_file).unlink()

        return exps
    else:
        return None

    
def save_params(path, params):
    params = {key: value.cpu() for key, value in params.items()}    
    torch.save(params, f'{path}/params.pt')
    open(f'{path}/sent_to_disk', 'a').close()
    

def load_params(path):
    if os.listdir(path):
        files = os.listdir(path)
        if 'sent_to_disk' in files:
            try:
                params = torch.load(f'{path}/params.pt', weights_only=True)
                print('Params loaded')
            except FileNotFoundError:
                print('File appeared to have disappeared')
        else:
            params = None
        for file in os.listdir(path):
            full_path = os.path.join(path, file)
            Path(full_path).unlink()

        return params

    else:
        return None


class LimitedQueue:
    def __init__(self, shape, max_size=8):
        self.queue = deque(maxlen=max_size)
        for i in range(max_size):
            self.queue.append(np.zeros(shape, dtype=np.float32))            

    def add(self, item):
        self.queue.append(item)

    def get_items(self):
        return np.stack(self.queue)



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
        


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
