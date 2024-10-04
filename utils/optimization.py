import torch
from torch.optim import Adam
import copy
import torch.nn as nn
import time



def Adam_update(params: list[dict],
                losses: list[float],
                keys: list[str],
                optimizers: list):

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


def reset_params(agent, params, optimizers, keys, lr):
    for key in keys:
        for name, param in params[key].items():
            if 'bias' in name:
                init = torch.nn.init.constant_(param, 0.0)
            else:
                init = torch.nn.init.xavier_normal_(param)
            params[key][name] = nn.Parameter(init)

        optimizers[key] = Adam(params[key].values(), lr)
    
    agent.log_alpha_skill = torch.tensor(agent.log_alpha_skill.item(), dtype=torch.float16,
                                          requires_grad=True, device=agent.device)
    agent.optimizer_alpha_skill = Adam([agent.log_alpha_skill], lr=lr)

    return params, optimizers, agent

    
