import os
import numpy as np

import pathlib


from segmentation_models.datasets.ply import dict2ply, read_ply
from segmentation_models.datasets.pts import read_pts

from open3d.ml.contrib  import subsample
import open3d

__all__ = ['preprocessing']

def preprocessing(file_path, label_to_color, label_encoder=None, grid_size = 0.0, processed_dir = None):

  ext = os.path.split(file_path)[-1].split('.')[-1]
  if ext == 'pts':
    is_pts = True
  elif ext == 'ply':
    is_pts = False
  else:
    raise RuntimeError('Unknown pointcloud file format')

  data_pts = read_pts(file_path)  if is_pts else read_ply(file_path)

  cloud = open3d.io.read_point_cloud(file_path)
  mean, cov = cloud.compute_mean_and_covariance()
  coords = data_pts[:, :3].astype(np.float32)
  means = np.mean(coords, axis=0)
  coords = coords - mean
  labels = data_pts[:, -1].astype(np.long)

  labels = label_encoder(labels) if label_encoder is not None else labels

  coords = coords[::grid_size]
  labels = labels[::grid_size]
  #if grid_size > 1e-02:
  #  coords, labels = subsample(coords, classes= labels, sampleDl=grid_size)
  labels = labels[:, np.newaxis]
  colors = label_to_color(labels)
  data_ply = { "x":     coords[:, 0],
               "y":     coords[:, 1],
               "z":     coords[:, 2],
               "red":   colors[:, 0],
               "green": colors[:, 1],
               "blue":  colors[:, 2]}


  path_ply = os.path.split(file_path)[-1].split(".")[-2] + '.ply'
  coords_npy = 'coords.npy'
  labels_npy = 'labels.npy'
  if processed_dir != None:
    pathlib.Path(processed_dir).mkdir(parents=True, exist_ok=True)
    path_ply = os.path.join(processed_dir, path_ply)
    coords_npy = os.path.join(processed_dir, coords_npy)
    labels_npy = os.path.join(processed_dir, labels_npy)

  if dict2ply(data_ply, path_ply):
      print('PLY point cloud successfully saved to {}'.format(path_ply))

  with open(coords_npy,'wb') as f:
      np.save(f, coords)

  with open(labels_npy, 'wb') as f:
      np.save(f,labels)

