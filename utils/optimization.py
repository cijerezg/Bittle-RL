import torch
from torch.optim import Adam
import copy
import torch.nn as nn
import time


def Adam_update(params: list[dict],
                losses: list[float],
                keys: list[str],
                optimizers: list,
                lr: float):

    for loss, key in zip(losses, keys):
        optimizers[key].zero_grad()
        loss.backward(retain_graph=True)
        optimizers[key].step()

    return params



def set_optimizers(params, keys, lr):
    optimizers = {}
    for key in keys:
        if 'skills' in key:
            parameters = (*params['Encoder'].values(),
                          *params['Decoder'].values())
        elif 'state' in key:
            parameters = (*params['StateEncoder'].values(),
                          *params['StateDecoder'].values())

        else:
            parameters = params[key].values()
                

        optimizers[key] = Adam(parameters, lr=lr)

    return optimizers