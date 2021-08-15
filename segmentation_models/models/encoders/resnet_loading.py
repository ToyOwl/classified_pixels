from .resnet import BasicBlock, BottleNeck, ResNet
import os
import torch
import inspect
from torchvision.models.utils import load_state_dict_from_url

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18' : 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34' : 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50' : 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
}

def pathToModel(relativePath):
    return os.path.abspath(os.path.join(os.path.dirname(inspect.getfile(inspect.currentframe())), relativePath))

model_weights = {
       'resnet18' : pathToModel('../weights/resnet18-5c106cde.pth'),
       'resnet34' : pathToModel('../weights/resnet34-333f7ec4.pth'),
       'resnet50' : pathToModel('../weights/resnet50-19c8e357.pth'),
       'resnet101': pathToModel('../weights/resnet101-5d3b4d8f.pth'),
       'resnet152': pathToModel('../weights/resnet152-b121ed2d.pth')
}

def resnet18(dilations, use_locals = True, **kwargs):
    return load_resnet('resnet18', BasicBlock, [2, 2, 2, 2], dilations, use_locals, **kwargs)

def resnet34(dilations, use_locals = True, **kwargs):
    return load_resnet('resnet34', BasicBlock, [3, 4, 6, 3], dilations, use_locals, **kwargs)

def resnet50(dilations, use_locals = True, **kwargs):
    return load_resnet('resnet50', BottleNeck, [3, 4, 6, 3], dilations, use_locals, **kwargs)

def resnet101(dilations, use_locals = True, **kwargs):
    return load_resnet('resnet101', BottleNeck, [3, 4, 23, 3], dilations, use_locals, **kwargs)

def resnet152(dilations, use_locals = True, **kwargs):
    return load_resnet('resnet152', BottleNeck, [3, 8, 36, 3], dilations, use_locals, **kwargs)


def load_resnet(arch, block, layers, dilations, use_locals = True, **kwargs):
    model = ResNet(block, layers,   **kwargs)
    if not use_locals:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=True)
    else :
        state_dict = torch.load(model_weights[arch])
    model.load_state_dict(state_dict)
    return model
