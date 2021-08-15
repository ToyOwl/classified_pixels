import torch
import numpy as np
import open3d

try:
  from torch_points_kernels import knn
  USING_TORCH_POINTS_KERNELS = True
except(ModuleNotFoundError, ImportError):
  USING_TORCH_POINTS_KERNELS = False



class PointCloudDataset:

  def __init__(self, points, device , n_decimation, n_neighbors):
    self.points = points
    self.n_layers  = len(n_decimation)
    self.n_decimations = n_decimation
    self.n_neighbors = n_neighbors
    self.device = device

  def prep_cloud(self):

    flat_inputs = self.prep_cloud_impl(self.points)

    inputs={}
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
    return inputs

  def prep_cloud_impl(self, point_cloud):

     features= point_cloud

     input_points = []
     input_neighbors = []
     input_pools = []
     input_up_samples = []

     for i in self.n_decimations:
       if USING_TORCH_POINTS_KERNELS:
         neighbour_idx, _ = knn(torch.from_numpy(point_cloud).cpu().contiguous(),
                                torch.from_numpy(point_cloud).cpu().contiguous(), self.n_neighbors)
         neighbour_idx = neighbour_idx.cpu().numpy()
       else:
         neighbour_idx = self.knn_cpu(point_cloud, point_cloud, self.n_neighbors)

       sub_points = point_cloud[:, :point_cloud.shape[1] // i, :]
       pool_i = neighbour_idx[:, :point_cloud.shape[1] // i, :]

       if USING_TORCH_POINTS_KERNELS:
         up_i, _ = knn(torch.from_numpy(sub_points).cpu().contiguous(),
                       torch.from_numpy(point_cloud).cpu().contiguous(), 1)
         up_i = up_i.cpu().numpy()
       else:
         up_i = self.knn_cpu(sub_points, point_cloud, 1)

       input_points.append(point_cloud)
       input_neighbors.append(neighbour_idx)
       input_pools.append(pool_i)
       input_up_samples.append(up_i)
       point_cloud = sub_points

     input_list = input_points + input_neighbors + input_pools + input_up_samples
     input_list += [features, np.full((features.shape[0], features.shape[1], 1), 255)]

     return input_list

  @classmethod
  def color_decode_target(cls, target):
    out = np.zeros(target.shape+(3,), dtype=np.uint8)
    for target_class in cls.classes:
       msk = np.where(target == target_class.train_id)
       out[msk] = cls.train_id_to_color[target_class.train_id]
    return out.squeeze()

  def knn_cpu(self, test_pc, query_pc, n_points):
    batch_idx = []
    for idx in range(test_pc.shape[0]):
       nns = open3d.core.nns.NearestNeighborSearch(open3d.core.Tensor.from_numpy(test_pc[idx,:,:]))
       nns.knn_index()
       neighbour_idx, _ = nns.knn_search(open3d.core.Tensor.from_numpy(query_pc[idx,:,:]), n_points)
       batch_idx.append(neighbour_idx.cpu().numpy())
    batch_idx = np.array(batch_idx)
    return batch_idx
