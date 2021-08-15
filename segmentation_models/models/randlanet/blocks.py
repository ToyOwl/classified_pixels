import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models.models.utils.layers import *

class LocalFeatureAggregation(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()

        self.mlp1 = conv_layer2d(d_in, d_out // 2, kernel_size=(1, 1), bn=True)
        self.lse =  LocalSpatialEncoding(d_out)
        self.mlp2 = conv_layer2d(d_out, d_out * 2, kernel_size=(1, 1), bn=True, activation=None)
        self.shortcut =conv_layer2d(d_in, d_out * 2, kernel_size=(1, 1), bn=True, activation=None)

    def forward(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)
        f_pc = self.lse(xyz, f_pc, neigh_idx)
        f_pc = self.mlp2(f_pc)
        shortcut = self.shortcut(feature)
        return F.leaky_relu(f_pc+shortcut, negative_slope=0.2)

class LocalSpatialEncoding(nn.Module):

    def __init__(self, d_out):
        super().__init__()

        self.mlp1 = conv_layer2d(10, d_out // 2, kernel_size=(1, 1), bn=True)
        self.mlp2 = conv_layer2d(d_out // 2, d_out // 2, kernel_size=(1, 1), bn=True)

        self.pool1 = AttentivePooling(d_out, d_out // 2)
        self.pool2 = AttentivePooling(d_out, d_out)

    def forward(self, coords, feature, indices):
        r"""
        Parameters
        ----------
        coords:   (B, N, point_dim)
        feature: (B,3,N, 1)
        Returns
        -------
        (B,  N,  n_neighbours, 10)
        """

        f_coords = self.position_encoding(coords, indices)
        f_coords = f_coords.permute((0, 3, 1, 2))
        f_coords = self.mlp1(f_coords)
        f_neighbours = self.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)), indices)
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))

        f_concat = torch.cat([f_neighbours, f_coords], dim=1)
        f_pool_aggregation = self.pool1(f_concat)

        f_coords = self.mlp2(f_coords)
        f_neighbours = self.gather_neighbour(f_pool_aggregation.squeeze(-1).permute((0, 2, 1)), indices)
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))
        f_concat = torch.cat([f_neighbours, f_coords], dim=1)
        f_pool_aggregation = self.pool2(f_concat)

        return f_pool_aggregation

    def position_encoding(self, coords, neigh_idx):
        r"""
        Parameters
        ----------
        coords:   (B, N, point_dim)
        neigh_idx: ()
        Returns
        -------
        (B,  N,  n_neighbours, 10)
        """
        neghbour_coords = self.gather_neighbour(coords, neigh_idx)
        coords_tile = coords.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)
        relative_coords = coords_tile - neghbour_coords
        relative_distances = torch.sqrt(torch.sum(torch.pow(relative_coords, 2), dim=-1, keepdim=True))
        out_feature = torch.cat([relative_distances, relative_coords, coords_tile, neghbour_coords], dim=-1)
        return out_feature

    @staticmethod
    def gather_neighbour(points, indices):
        r"""
        Parameters
        ----------
        features:   (B, N, point_dim)
        indices:    ()
        Returns
        -------
        (B,  point_dim, N,  n_neighbours)
        """
        batch_size = points.shape[0]
        num_points = points.shape[1]
        d = points.shape[2]
        index_input = indices.reshape(batch_size, -1)
        features = torch.gather(points, 1, index_input.unsqueeze(-1).repeat(1, 1, points.shape[2]))
        features = features.reshape(batch_size, num_points, indices.shape[-1], d)
        return features

class AttentivePooling(nn.Module):

  def __init__(self, d_in, d_out):
    super().__init__()
    self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
    self.mlp = conv_layer2d(d_in, d_out, kernel_size=(1, 1), bn=True)

  def forward(self, features):
    r"""
    Parameters
    ----------
    features:  shape (B, d_in, N, K)

    Returns
    -------
    (B, d_out, N, 1)
    """

    activation = self.fc(features)
    scores = F.softmax(activation, dim=3)
    aggregations = features * scores
    aggregations = torch.sum(aggregations, dim=3, keepdim=True)
    aggregations = self.mlp(aggregations)
    return aggregations
