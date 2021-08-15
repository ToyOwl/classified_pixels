import torch
import os
import open3d
import numpy as np

from collections import namedtuple
from pyntcloud import PyntCloud

from .preprocessing import preprocessing
from .utils import top_view, write_point_cloud, augmentation_transform
from .RandLaNetDataset import RandLaNetDataset


from torch.utils.data import DataLoader

def preprocessing(path_to_dir, encoder, decoder, subsampling = 10):
    clouds_coords, clouds_labels, trees = [], [], []
    for (root, dirs, files) in os.walk(path_to_dir):
        for file in files:
           cloud = PyntCloud.from_file(os.path.join(path_to_dir, file))
           labels = np.array(cloud.points['class'])
           labels = labels.astype(np.long)
           labels = encoder(labels)
           labels = labels[::subsampling]
           coords = cloud.xyz[::subsampling]
           cloud  = open3d.geometry.PointCloud()
           cloud.points = open3d.utility.Vector3dVector(coords)
           mean, _ = cloud.compute_mean_and_covariance()
           cloud.points = open3d.utility.Vector3dVector(coords - mean)
           clouds_coords += [np.asarray(cloud.points).astype(np.float32)]
           clouds_labels += [labels]
           trees  += [open3d.geometry.KDTreeFlann(cloud)]
           write_point_cloud(coords, decoder(labels), file)
    return clouds_coords, clouds_labels, trees

class DaLesDataset(RandLaNetDataset):

  DaLesClass = namedtuple('DaLesDataset', ['name', 'id', 'train_id', 'color', 'train_color'])

  classes = [
             DaLesClass('clutter',     0, 3, (000, 000, 139), (000, 000, 139)),
             DaLesClass('groud',       1, 0, (000, 000, 255), (000, 000, 255)),
             DaLesClass('vegetation',  2, 2, (000, 255, 000), (000, 255, 000)),
             DaLesClass('cars',        3, 3, (255, 000, 255), (000, 000, 139)),
             DaLesClass('trucks',      4, 3, (255, 255, 000), (000, 000, 139)),
             DaLesClass('power lines', 5, 3, (152, 251, 152), (000, 000, 139)),
             DaLesClass('fences',      6, 3, (173, 216, 230), (000, 000, 139)),
             DaLesClass('poles',       7, 3, (255, 69, 000),  (255, 000, 000)),
             DaLesClass('buildings',   8, 1, (255, 000, 000), (255, 000, 000)),
]

  num_features = 3
  num_classes = 4
  ignore_index = 255
  class_to_names = {0: 'groud', 1: 'vegetation', 2: 'buildings', 3: 'clutter' }

  _, train_unique_ids  = np.unique([c.train_id for c in classes ], return_index=True)
  train_id_to_color =[]

  for idx in train_unique_ids:
    if classes[idx].train_id != 255:
      train_id_to_color.append(classes[idx].train_color)

  train_id_to_color.append([0, 0, 0])
  train_id_to_color = np.array(train_id_to_color)
  id_to_train_id = np.array([c.train_id for c in classes])


  def __init__(self, clouds_coords, clouds_labels, trees,  device , n_points, n_decimation, n_neighbors, sigma=0.35, augmentation= None):
      RandLaNetDataset.__init__(self, clouds_coords=clouds_coords, clouds_labels=clouds_labels,
                                trees=trees, device=device, n_points=n_points, n_decimation=n_decimation,
                                n_neighbors=n_neighbors,
                                sigma=sigma, augmentation=augmentation)


def get_dales_dataloaders(pth_train_dataset, pth_val_dataset, device, n_batches, n_points, n_decimiations,
                          n_neighboors, n_sigma=10.5, subsampling=10):

    clouds_coords, clouds_labels, trees = preprocessing(pth_train_dataset, DaLesDataset.encode_target,
                                                        DaLesDataset.color_encode_target, subsampling=subsampling)

    val_clouds_coords, val_clouds_labels, val_trees = preprocessing(pth_val_dataset, DaLesDataset.encode_target,
                                                        DaLesDataset.color_encode_target, subsampling=subsampling)

    train_dataset = DaLesDataset(clouds_coords, clouds_labels, trees, device,n_points,
                                 n_decimiations, n_neighboors, n_sigma, augmentation=augmentation_transform)
    val_dataset = DaLesDataset(val_clouds_coords, val_clouds_labels, val_trees, device, n_points,
                               n_decimiations, n_neighboors, n_sigma, augmentation=augmentation_transform)

    return  DataLoader(train_dataset, n_batches, collate_fn=train_dataset.collate_fn), \
            DataLoader(val_dataset, n_batches, collate_fn=val_dataset.collate_fn), \
            train_dataset.num_features, train_dataset.num_classes, train_dataset.class_to_names, \
            top_view(color_decoder=DaLesDataset.color_encode_target, size=(256, 256))


