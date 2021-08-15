from .segmenation_model import *
from .deeplabv3.deeplabv3 import DeepLabV3, DeepLabV3Plus
from .encoders.resnet import *
from .encoders.resnet_loading import *
from .utils.constants import*
from .segmenation_model import SegmentationModel
from torch import nn
from collections import OrderedDict

__all__ = ['deeplab_map', 'get_segmenation_model']

def get_deeplab_resnet(num_classes, deeplab_type, resnet_type, in_features, ll_features=256, out_stride=8, bn_momentum=0.01):
  if out_stride == 8:
    dilations =  True
    aspp_dilate = [12, 24, 36]
  else:
    dilations = False
    aspp_dilate = [6, 12, 18]

  if deeplab_type == 'deeplabv3':
    return_layers = {'layer4': 'out'}
    classifier = DeepLabV3(in_features, num_classes, aspp_dilate)
  elif deeplab_type == 'deeplabv3plus':
    return_layers = {'layer4': 'out', 'layer1': 'low_level'}
    classifier = DeepLabV3Plus(in_features, ll_features, num_classes, aspp_dilate)
  else:
    raise RuntimeError('unknown architecture type')

  if resnet_type not in resnet_map:
    raise RuntimeError('unknown backbone type')
  else:
   backbone = resnet_map[resnet_type](dilations)

  backbone = DictLayersGetter(backbone, out_layers=return_layers)

  #for module in backbone.modules():
  #    if isinstance(module, nn.BatchNorm2d):
  #        module.momentum = bn_momentum

  model = SegmentationModel(backbone, classifier)

  return model

def deeplabv3_resnet18(num_classes, out_stride =8, bn_momentum=0.1):
    return get_deeplab_resnet(num_classes, 'deeplabv3', 'resnet18', 512, out_stride=out_stride, bn_momentum=bn_momentum)

def deeplabv3_resnet34(num_classes, out_stride=8, bn_momentum=0.1):
  return get_deeplab_resnet(num_classes, 'deeplabv3', 'resnet34',  512, out_stride=out_stride, bn_momentum=bn_momentum)

def deeplabv3_resnet50(num_classes, out_stride=8, bn_momentum=0.1):
  return get_deeplab_resnet(num_classes, 'deeplabv3', 'resnet50', 2048, out_stride=out_stride, bn_momentum=bn_momentum)

def deeplabv3plus_resnet50(num_classes, out_stride=8, bn_momentum=0.1):
  return get_deeplab_resnet(num_classes, 'deeplabv3plus', 'resnet50', 2048, out_stride=out_stride, bn_momentum=bn_momentum)

def deeplabv3_resnet101(num_classes, out_stride=8, bn_momentum=0.1):
  return get_deeplab_resnet(num_classes, 'deeplabv3', 'resnet101', 2048, out_stride=out_stride, bn_momentum=bn_momentum)

def deeplabv3plus_resnet101(num_classes, out_stride=16, bn_momentum=0.1):
  return get_deeplab_resnet(num_classes, 'deeplabv3plus', 'resnet101', 2048, out_stride=out_stride, bn_momentum=bn_momentum)

def deeplabv3_resnet152(num_classes, out_stride=16, bn_momentum=0.1):
  return get_deeplab_resnet(num_classes, 'deeplabv3', 'resnet152', 2048, out_stride=out_stride, bn_momentum=bn_momentum)

def deeplabv3plus_resnet152(num_classes, out_stride=8, bn_momentum=0.1):
  return get_deeplab_resnet(num_classes, 'deeplabv3plus', 'resnet152', 2048, out_stride=out_stride, bn_momentum=bn_momentum)

resnet_map = {'resnet18' : resnet18,
              'resnet34' : resnet34,
              'resnet50' : resnet50,
              'resnet101': resnet101,
              'resnet152': resnet152}

deeplab_map = { 'deeplabv3-resnet18'       : deeplabv3_resnet18,
                'deeplabv3-resnet34'       : deeplabv3_resnet34,
                'deeplabv3-resnet50'       : deeplabv3_resnet50,
                'deeplabv3plus-resnet50'   : deeplabv3plus_resnet50,
                'deeplabv3-resnet101'      : deeplabv3_resnet101,
                'deeplabv3plus-resnet101'  : deeplabv3plus_resnet101,
                'deeplabv3-resnet152'      : deeplabv3_resnet152,
                'deeplabv3plus-resnet152'  : deeplabv3plus_resnet152}

def get_segmenation_model(model_name, num_classes, out_stride = 8,bn_momentum=0.1):
    return deeplab_map[model_name](num_classes, out_stride, bn_momentum)

class DictLayersGetter(nn.ModuleDict):

  def __init__(self, model, out_layers):

    if not set(out_layers).issubset([name for name, _ in model.named_children()]):
       raise ValueError("can't find return_layers in the model")

    model_layers = out_layers
    out_layers = {k: v for k, v in out_layers.items()}
    layers = OrderedDict()
    for name, module in model.named_children():
      layers[name] = module
      if name in out_layers:
        del out_layers[name]
      if not out_layers:
        break

    super(DictLayersGetter, self).__init__(layers)
    self.out_layers = model_layers

  def forward(self, x):
    out = OrderedDict()
    for name, module in self.named_children():
      x = module(x)
      if name in self.out_layers:
        out_name = self.out_layers[name]
        out[out_name] = x
    return out
