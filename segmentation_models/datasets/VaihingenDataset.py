import torch
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
from collections import namedtuple

from .preprocessing import  preprocessing
from .utils import top_view

import open3d

try:
  from torch_points_kernels import knn
  USING_TORCH_POINTS_KERNELS = True
except(ModuleNotFoundError, ImportError):
  USING_TORCH_POINTS_KERNELS = False

weights =[]

class VaihingenDataset(Dataset):

  VaihingenClass = namedtuple('VaihingenDataset', ['name', 'id', 'train_id', 'color'])

  classes = [ VaihingenClass('powerline',           0,  3,     (255, 000, 255)),
              VaihingenClass('low_vegetation',      1,  0,     (000, 000, 255)),
              VaihingenClass('impervious_surfaces', 2,  0,     (000, 000, 255)),
              VaihingenClass('car',                 3,  3,     (255,  000, 255)),
              VaihingenClass('fence_hedge',         4,  3,     (255,  000, 255)),
              VaihingenClass('roof',                5,  1,     (255, 000, 000)),
              VaihingenClass('facade',              6,  1,     (255, 000, 000)),
              VaihingenClass('shrub',               7,  2,     (000, 255, 000)),
              VaihingenClass('tree',                8,  2,     (000, 255, 000))]

  num_features =3
  num_classes = 4
  ignore_index = 255

  _, train_unique_ids  = np.unique([c.train_id for c in classes ], return_index=True)
  train_id_to_color =[]

  for idx in train_unique_ids:
    if classes[idx].train_id != 255:
      train_id_to_color.append(classes[idx].color)

  train_id_to_color.append([0, 0, 0])
  train_id_to_color = np.array(train_id_to_color)
  id_to_train_id = np.array([c.train_id for c in classes])

  def __init__(self, path_to_dataset, device , n_points, n_decimation, n_neighbors, sigma=0.35):
    self.cloud = np.load(os.path.join(path_to_dataset,'coords.npy'))
    self.labels = np.load(os.path.join(path_to_dataset,'labels.npy'))

    if self.cloud.shape[0] != self.labels.shape[0]:
        raise ValueError('number of points and labels do not match')

    unique, counts = np.unique(self.labels, return_counts=True)

    self.weights = torch.tensor(counts[:self.num_classes], dtype=torch.float32, device=device)
    self.weights = self.weights/torch.tensor(counts[:self.num_classes], dtype=torch.float32, device=device).sum()
    self.weights = 1/(self.weights + 2e-3)

    self.pcd = open3d.geometry.PointCloud()
    self.pcd.points = open3d.utility.Vector3dVector(self.cloud)
    self.tree =open3d.geometry.KDTreeFlann(self.pcd)
    self.possibility = np.random.randn(self.cloud.shape[0])*1e-03
    self.n_layers = len(n_decimation)
    self.n_decimations = n_decimation
    self.n_points = n_points
    self.n_neighbors = n_neighbors
    self.sigma  = sigma
    self.device = device


  def __len__(self):
    return len(self.cloud) //self.n_points

  def __getitem__(self, item):
    pick_idx = np.argmin(self.possibility)
    points, labels, idx, select_point = self.crop_pc(self.cloud, self.labels, self.tree, pick_idx, self.n_points)
    dists = np.sum(np.square((self.cloud[idx] - select_point).astype(np.float32)), axis=1)
    delta = np.square(1 - dists / np.max(dists))
    self.possibility[idx] += delta
    return  points.astype(np.float32), labels.astype(np.int32), idx.astype(np.int32)

  def crop_pc(self, points, labels, search_tree, pick_idx, num_points):
    center_point = points[pick_idx,:].reshape(-1,1)
    noise = np.random.normal(scale = self.sigma, size=center_point.shape)
    center_point = center_point + noise.astype(center_point.dtype)
    [k,select_idx, _] = search_tree.search_knn_vector_3d(center_point.astype(np.float64), num_points)
    center_point = np.transpose(center_point, (1,0))
    select_idx = np.asarray(select_idx)
    select_idx = self.shuffle_idx(select_idx)
    select_points = points[select_idx]
    select_labels = labels[select_idx]
    return select_points, select_labels, select_idx, center_point.astype(np.float32)

  @classmethod
  def shuffle_idx(cls, x):
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    return x[idx]

  @classmethod
  def encode_target(cls, target):
    return cls.id_to_train_id[np.array(target)]

  @classmethod
  def color_encode_target(cls, target):
    out = np.zeros(target.shape+(3,), dtype=np.uint8)
    for target_class in cls.classes:
       msk = np.where(target == target_class.train_id)
       if target_class.train_id != 255:
         out[msk]  = cls.train_id_to_color[target_class.train_id]
    return out.squeeze()

  @classmethod
  def decode_target(cls, target):
    out = np.zeros(target.shape+(3,), dtype=np.uint8)
    for target_class in cls.classes:
       msk = np.where(target == target_class.id)
       target_id = cls.id_to_train_id[target_class.id]
       if target_id != 255:
         out[msk]  = cls.train_id_to_color[target_id]

    return out.squeeze()

  def tf_map(self, batch_pc, batch_label, batch_pc_idx):
     features = batch_pc
     input_points = []
     input_neighbors = []
     input_pools = []
     input_up_samples = []

     for i in self.n_decimations:
       if USING_TORCH_POINTS_KERNELS:
         print('downsampling  {} points from {} with  {} Nans'.format(self.n_neighbors, batch_pc.shape[1],np.count_nonzero(np.isnan(batch_pc)) ))
         neighbour_idx, _ = knn(torch.from_numpy(batch_pc).cpu().contiguous().type(torch.float32),
                           torch.from_numpy(batch_pc).cpu().contiguous().type(torch.float32), self.n_neighbors)
         neighbour_idx = neighbour_idx.cpu().numpy()
       else:
         neighbour_idx =   self.knn_cpu(batch_pc, batch_pc, self.n_neighbors)

       sub_points = batch_pc[:, :batch_pc.shape[1] // i, :]
       pool_i = neighbour_idx[:, :batch_pc.shape[1] // i, :]

       if USING_TORCH_POINTS_KERNELS:
         up_i, _ = knn(torch.from_numpy(sub_points).cpu().contiguous().type(torch.float32),
                       torch.from_numpy(batch_pc).cpu().contiguous().type(torch.float32), 1)
         up_i    = up_i.cpu().numpy()
       else:
         up_i =   self.knn_cpu(sub_points, batch_pc, 1)

       input_points.append(batch_pc)
       input_neighbors.append(neighbour_idx)
       input_pools.append(pool_i)
       input_up_samples.append(up_i)
       batch_pc = sub_points

     input_list = input_points + input_neighbors + input_pools + input_up_samples
     input_list += [features, batch_label, batch_pc_idx]

     return input_list

  def collate_fn(self,batch):

    selected_pc, selected_labels, selected_idx= [],[],[]

    for i in range(len(batch)):
       selected_pc.append(batch[i][0])
       selected_labels.append(batch[i][1])
       selected_idx.append(batch[i][2])

    selected_pc = np.stack(selected_pc)
    selected_labels = np.stack(selected_labels)
    selected_idx = np.stack(selected_idx)


    flat_inputs = self.tf_map(selected_pc, selected_labels, selected_idx)

    inputs = {}
    inputs['xyz'] = []

    for tmp in flat_inputs[:self.n_layers]:
        inputs['xyz'].append(torch.from_numpy(tmp).float().to(self.device))

    inputs['neigh_idx'] = []
    for tmp in flat_inputs[self.n_layers: 2 * self.n_layers]:
      inputs['neigh_idx'].append(torch.from_numpy(tmp).long().to(self.device))

    inputs['sub_idx'] = []
    for tmp in flat_inputs[2 * self.n_layers:3 * self.n_layers]:
        inputs['sub_idx'].append(torch.from_numpy(tmp).long().to(self.device))

    inputs['interp_idx'] = []
    for tmp in flat_inputs[3 * self.n_layers:4 * self.n_layers]:
        inputs['interp_idx'].append(torch.from_numpy(tmp).long().to(self.device))

    inputs['features'] = torch.from_numpy(flat_inputs[4 * self.n_layers]).transpose(1,2).float().to(self.device)
    inputs['labels'] = torch.from_numpy(flat_inputs[4 * self.n_layers + 1]).long().to(self.device)
    inputs['input_inds'] = torch.from_numpy(flat_inputs[4 * self.n_layers + 2]).long().to(self.device)

    return inputs

  def knn_cpu(self, test_pc, query_pc, n_points):
    batch_idx = []
    for idx in range(test_pc.shape[0]):
       nns = open3d.core.nns.NearestNeighborSearch( open3d.core.Tensor.from_numpy(test_pc[idx,:,:]))
       nns.knn_index()
       neighbour_idx, _ = nns.knn_search(open3d.core.Tensor.from_numpy(query_pc[idx,:,:]), n_points)
       batch_idx.append(neighbour_idx.cpu().numpy())
    batch_idx = np.array(batch_idx)
    return batch_idx

def get_vaihingen_dataloaders(pth_train_dataset, pth_val_dataset, device, n_batches, n_points, n_decimiations,
                              n_neighboors, n_sigma=10.5, subsampling=1):

    preprocessing(os.path.join(pth_train_dataset, 'traininig.pts'), label_to_color=VaihingenDataset.color_encode_target,
                  label_encoder=VaihingenDataset.encode_target,
                  grid_size=subsampling, processed_dir=pth_train_dataset)

    train_dataset = VaihingenDataset(pth_train_dataset, device, n_points, n_decimiations, n_neighboors, n_sigma)
    val_dataset = VaihingenDataset(pth_train_dataset, device, n_points, n_decimiations, n_neighboors, n_sigma)

    return  DataLoader(train_dataset, n_batches, collate_fn=train_dataset.collate_fn), \
            DataLoader(val_dataset, n_batches, collate_fn=val_dataset.collate_fn), \
            train_dataset.num_features, train_dataset.num_classes, train_dataset.class_to_names, \
            top_view(color_decoder =VaihingenDataset.color_encode_target, size=(256, 256))

