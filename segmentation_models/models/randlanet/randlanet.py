import torch
import torch.nn as nn

from segmentation_models.models.utils.layers import *
from segmentation_models.models.randlanet.blocks import LocalFeatureAggregation, LocalSpatialEncoding, AttentivePooling

import numpy as np
from sklearn.metrics import confusion_matrix


class RandLANet(nn.Module):

  model_name = 'RandLANet'

  def __init__(self, n_features, n_classes, dim_out):

    super().__init__()

    d_in = 8
    self.fc_start = conv_layer1d(n_features, d_in, kernel_size=1, bn=True)
    self.encoder = nn.ModuleList()

    self.n_layers =len(dim_out)

    for i in range(self.n_layers):
        d_out = dim_out[i]
        self.encoder.append(LocalFeatureAggregation(d_in, d_out))
        d_in = 2 * d_out

    d_out = d_in
    self.dc_start = conv_layer2d(d_in, d_out, kernel_size=(1, 1), bn=True)
    self.decoder = nn.ModuleList()

    for idx in range(self.n_layers):
       if idx < self.n_layers -1:
         d_in = d_out + 2 * dim_out[-idx-2]
         d_out = 2 * dim_out[-idx-2]
       else:
         d_in = 4 * dim_out[-self.n_layers]
         d_out = 2 * dim_out[-self.n_layers]
       self.decoder.append(conv_layer2d(d_in, d_out, kernel_size=(1, 1), bn=True))

    self.fc_end = nn.Sequential(conv_layer2d(d_out, 64, kernel_size=(1, 1), bn=True),
                                conv_layer2d(64, 32, kernel_size=(1, 1), bn=True),
                                nn.Dropout(0.5),
                                conv_layer2d(32, n_classes, kernel_size=(1, 1), bn=False, activation=None))

  def forward(self, input):
    r"""
    Parameters
    ----------
    input: dict
    input['features']   (B, N, d_in) point cloud

    Returns
    -------
    dict ,
    dict['scores '] (B, num_classes, N) segmentation scores for each point
    """
    features = input['features']
    features = self.fc_start(features)
    features = features.unsqueeze(dim=3)

    #ENCODER
    encoder_list = []
    for idx in range(self.n_layers):
       x = self.encoder[idx](features, input['xyz'][idx], input['neigh_idx'][idx])
       f_sampled_i = self.random_sample(x, input['sub_idx'][idx])
       features = f_sampled_i
       if idx == 0:
          encoder_list.append(x)
       encoder_list.append(f_sampled_i)

    features = self.dc_start(encoder_list[-1])

    #DECODER
    dc_list = []
    for j in range(self.n_layers):
       f_interp_i = self.nearest_interpolation(features, input['interp_idx'][-j - 1])
       f_decoder_i = self.decoder[j](torch.cat([encoder_list[-j - 2], f_interp_i], dim=1))
       features = f_decoder_i
       dc_list.append(f_decoder_i)

    features  =self.fc_end(features)
    out = features.squeeze(-1)

    out_dict ={}
    out_dict['logits'] = out
    out_dict['labels'] = input['labels'].squeeze(-1)
    out_dict['predicts'] = torch.max(out, dim=-2).indices

    return out_dict

  @staticmethod
  def random_sample(feature, pool_idx):
    r"""
    Parameters
    ----------
    feature:   (B, f_dim,  N, 1)
    pool_idx:  (B, pool_N, n_neighboors)

    Returns
    -------
    (B, f_dim,  N', 1)
    """
    feature = feature.squeeze(dim=3)
    num_neigh = pool_idx.shape[-1]
    d = feature.shape[1]
    batch_size = pool_idx.shape[0]
    pool_idx = pool_idx.reshape(batch_size, -1)
    pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
    pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
    pool_features = pool_features.max(dim=3, keepdim=True)[0]
    return pool_features

  @staticmethod
  def nearest_interpolation(feature, interp_idx):
     r"""
     Parameters
     ----------
     feature: (B, N, f_dim)
     coords:  (B, upsN, 1)
     Returns
     -------
     (B,  upsN, f_dim)
     """
     feature = feature.squeeze(dim=3)
     batch_size = interp_idx.shape[0]
     up_num_points = interp_idx.shape[1]
     interp_idx = interp_idx.reshape(batch_size, up_num_points)
     interpolated_features = torch.gather(feature, 2, interp_idx.unsqueeze(1).repeat(1,feature.shape[1],1))
     interpolated_features = interpolated_features.unsqueeze(3)
     return interpolated_features






