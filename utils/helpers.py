import torch
import copy

def get_params(model):
    params = {}
    for name, param in model.named_parameters():
        params[name] = nn.Parameter(param)
    return copy.deepcopy(params)
        
