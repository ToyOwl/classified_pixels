import torch
from torch import nn
from torch.nn import functional as F

from ..utils import atrous_spearable_convolution

__all__ = ['DeepLabV3', 'DeepLabV3Plus']

class ASPPConv(nn.Sequential):

  def __init__(self, in_channels, out_channels, dilation):
    modules = [nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
               nn.BatchNorm2d(out_channels), nn.ReLU()]
    super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):

  def __init__(self, in_channels, out_channels):
     super(ASPPPooling, self).__init__\
     (nn.AdaptiveAvgPool2d(1),nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels),nn.ReLU())

  def forward(self, x):
    size = x.shape[-2:]
    x = super(ASPPPooling, self).forward(x)
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):

    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabV3(nn.Module):

    def __init__(self, in_channels, num_classes, atrous_rates=[12, 24, 36]):
        super(DeepLabV3, self).__init__()
        self.classifier = nn.Sequential(ASPP(in_channels, atrous_rates), nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, num_classes, 1))
        self._initialize()

    def forward(self, feature):
        return self.classifier(feature['out'])

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

class DeepLabV3Plus(nn.Module):

    def __init__(self, in_channels, low_level_channels, num_classes, atrous_rates=[12, 24, 36], use_atrous=True):
        super(DeepLabV3Plus, self).__init__()
        self.project =  nn.Sequential(nn.Conv2d(low_level_channels, 48, 1, bias=False), nn.BatchNorm2d(48),
                          nn.ReLU(inplace=True))
        self.aspp = ASPP(in_channels, atrous_rates)
        self.classifier = nn.Sequential(
            atrous_spearable_convolution(304, 256, 3, padding=1, bias=False)
            if use_atrous else nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, num_classes, 1))
        self._initialize()

    def forward(self, feature):
        low_level_feature = self.project(feature['low_level'])
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:],
                                       mode='bilinear', align_corners=False)
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)



