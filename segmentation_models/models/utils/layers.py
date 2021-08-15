import torch.nn as nn

class conv_layer(nn.Sequential):

  def __init__( self, in_size, out_size, kernel_size, stride, padding, activation, bn, init, conv=None,
                                      batch_norm=None, bias=True, instance_norm=False, instance_norm_func=None):
     super().__init__()

     bias = bias and (not bn)

     conv_unit = conv(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

     init(conv_unit.weight)

     if bias:
        nn.init.constant_(conv_unit.bias, 0)

     self.add_module('conv', conv_unit)

     if bn:
       bn_unit = batch_norm(out_size)
       self.add_module('bn', bn_unit)

     if activation is not None:
       self.add_module('activation', activation)

     if not bn and instance_norm and instance_norm_func:
       in_unit = instance_norm_func(out_size, affine=False, track_running_stats=False)
       self.add_module('in', in_unit)

class conv_layer1d(conv_layer):

   def __init__(self, in_size, out_size, kernel_size=1, stride=1, padding=0,
                activation=nn.LeakyReLU(negative_slope=0.2, inplace=True), bn=False,  init=nn.init.kaiming_normal_, bias=True, instance_norm=False):
     super().__init__(in_size, out_size, kernel_size, stride, padding, activation, bn, init, conv=nn.Conv1d,
                         batch_norm=bn_layer1d, bias=bias, instance_norm=instance_norm, instance_norm_func=nn.InstanceNorm1d)

class conv_layer2d(conv_layer):

    def __init__(self, in_size, out_size, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                 activation=nn.LeakyReLU(negative_slope=0.2, inplace=True), bn=False, init=nn.init.kaiming_normal_,
                                                                                        bias=True, instance_norm=False):

      super().__init__(in_size, out_size, kernel_size, stride, padding, activation, bn, init, conv=nn.Conv2d,
                 batch_norm=bn_layer2d, bias=bias, instance_norm=instance_norm, instance_norm_func=nn.InstanceNorm2d)

class bn_layer(nn.Sequential):

  def __init__(self, in_size, batch_norm=None):
     super().__init__()
     self.add_module("bn", batch_norm(in_size, eps=1e-6, momentum=0.99))
     nn.init.constant_(self[0].weight, 1.0)
     nn.init.constant_(self[0].bias, 0)

class bn_layer1d(bn_layer):

  def __init__(self, in_size):
     super().__init__(in_size, batch_norm=nn.BatchNorm1d)

class bn_layer2d(bn_layer):

   def __init__(self, in_size):
      super().__init__(in_size, batch_norm=nn.BatchNorm2d)

class atrous_spearable_convolution(nn.Module):

   def __init__(self, in_size, out_size, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(atrous_spearable_convolution, self).__init__()
        self.separable_conv2d = nn.Conv2d(in_size, in_size, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=bias, groups=in_size)
        self.pointwise_conv2d= nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=bias)
        self._initialize(bias)

   def forward(self, x):
      return self.pointwise_conv2d(self.separable_conv2d(x))

   def _initialize(self, bias):
       nn.init.kaiming_normal(self.separable_conv2d.weight)
       nn.init.kaiming_normal(self.pointwise_conv2d.weight)
       if bias:
         nn.init.kaiming_normal(self.pointwise_conv2d.bias)
         nn.init.kaiming_normal(self.separable_conv2d.bias)
