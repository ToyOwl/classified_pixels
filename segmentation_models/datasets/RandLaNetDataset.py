import torch
import open3d
import numpy as np

from torch.utils.data import Dataset

try:
  from torch_points_kernels import knn
  USING_TORCH_POINTS_KERNELS = True
except(ModuleNotFoundError, ImportError):
  USING_TORCH_POINTS_KERNELS = False

class RandLaNetDataset(Dataset):

  def __init__(self, clouds_coords, clouds_labels, trees,  device , n_points, n_decimation, n_neighbors, sigma=0.35, augmentation=None):

    self.clouds = clouds_coords
    self.labels = clouds_labels
    self.trees  = trees
    self.possibility, self.min_possibility = [], []
    array_lbls = np.concatenate(self.labels).ravel()
    unique, counts = np.unique(array_lbls, return_counts=True)

    self.weights = torch.tensor(counts[:self.num_classes], dtype=torch.float32, device=device)
    self.weights = self.weights/torch.tensor(counts[:self.num_classes], dtype=torch.float32, device=device).sum()
    self.weights = 1/(self.weights + 2e-3)

    self.n_neighbors = n_neighbors
    self.n_decimations = n_decimation
    self.n_points  = n_points
    self.sigma = sigma
    self.n_layers = len(n_decimation)
    self.device = device
    self.augmentation = augmentation

    self.full_len =0

    for idx in range(len(self.clouds)):
        self.possibility += [np.random.rand(self.clouds[idx].shape[0]) * 1e-3]
        self.min_possibility += [float(np.min(self.possibility[-1]))]
        self.full_len +=self.clouds[idx].shape[0]

  def __len__(self):
    return self.full_len//self.n_points

  def __getitem__(self, item):
    cloud_ind = int(np.argmin(self.min_possibility))
    pick_idx = np.argmin(self.possibility[cloud_ind])
    points, labels, idx, select_point = \
        self.crop_pc(self.clouds[cloud_ind], self.labels[cloud_ind], self.trees[cloud_ind], pick_idx, self.n_points)
    dists = np.sum(np.square((self.clouds[cloud_ind][idx] - select_point).astype(np.float32)), axis=1)
    delta = np.square(1 - dists / np.max(dists))
    self.possibility[cloud_ind][idx] += delta
    self.min_possibility[cloud_ind] = np.min(self.possibility[cloud_ind])
    return  points.astype(np.float32), labels.astype(np.int32), idx.astype(np.int32)

  def crop_pc(self, points, labels, search_tree, pick_idx, num_points):

    center_point = points[pick_idx,:].reshape(-1,1)
    noise = np.random.normal(scale=self.sigma, size=center_point.shape)
    center_point = center_point + noise.astype(center_point.dtype)
    [k,select_idx, _] = search_tree.search_knn_vector_3d(center_point.astype(np.float64), num_points)
    center_point = np.transpose(center_point, (1,0))
    select_idx = np.asarray(select_idx)
    select_idx = self.shuffle_idx(select_idx)
    select_points = points[select_idx]

    if self.augmentation is not None:
        select_points = self.augmentation(select_points)

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
         out[msk] = cls.train_id_to_color[target_id]

    return out.squeeze()

  def tf_map(self, batch_pc, batch_label, batch_pc_idx):
     features = batch_pc
     input_points = []
     input_neighbors = []
     input_pools = []
     input_up_samples = []

     for i in self.n_decimations:
       if USING_TORCH_POINTS_KERNELS:
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
         up_i = up_i.cpu().numpy()
       else:
         up_i = self.knn_cpu(sub_points, batch_pc, 1)

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