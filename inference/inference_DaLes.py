import torch
import open3d

import os
import argparse
import numpy as np
from collections import namedtuple


from segmentation_models.models.randlanet.randlanet import RandLANet
from segmentation_models.datasets.utils import write_point_cloud

from pc_preprocessing import interpolate_dense_labels, threshold_predictions, eval_probs, load_open3d_cloud
from PointCloudDataset import PointCloudDataset


class DaLesPointCloudDataset(PointCloudDataset):

  BaseClass = namedtuple('DaLesPointCloudDatasetClass', ['name', 'train_id', 'color'])

  classes = [BaseClass('clutter ',   3, (255, 000, 255)),
             BaseClass('glo',        0, (000, 000, 255)),
             BaseClass('building',   1, (255, 000, 000)),
             BaseClass('vegetation', 2, (000, 255, 000))]

  n_classes = 4
  n_features = 3

  _, train_unique_ids  = np.unique([c.train_id for c in classes ], return_index=True)
  train_id_to_color =[]

  for idx in train_unique_ids:
    if classes[idx].train_id != 255:
      train_id_to_color.append(classes[idx].color)

  train_id_to_color.append([0, 0, 0])
  train_id_to_color = np.array(train_id_to_color)
  id_to_train_id = np.array([c.train_id for c in classes])

  def __init__(self, points, device , n_decimation, n_neighbors):
      PointCloudDataset.__init__(self, points=points, device=device,
                                 n_decimation=n_decimation, n_neighbors=n_neighbors)

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--point_cloud',   default=None,  type=str,    help='path to  images')
    parser.add_argument('--output_name',   default=None,  type=str,    help='path to decoded images')
    parser.add_argument('--model_path',    default=None,  type=str,    help='path to trained  model')

    parser.add_argument('--threshold',     default= .25,  type=float,   help='minimum confidence threshold')
    parser.add_argument('--subsampling',   default='y',   type=str,     help='using subsampling inference strategy', choices=['y', 'n'])
    parser.add_argument('--step',          default=2,     type=int,     help='random subsampling step')
    parser.add_argument('--dense_labels',  default='y',   type=str,     help='interpolate sparse output if subsampling True', choices=['y', 'n'])
    parser.add_argument('--knn_neighbors', default=5,     type=int,     help='interpolate knn neighbors')

    parser.add_argument('--n_blocks',    default=5, type=int, choices=[6, 5], help='RandLaNet blocks number')

    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    return parser



randla_net_blocks ={6: [16, 64, 128, 256, 512, 1024], 5: [16, 64, 128, 256, 512]}
decimations_blocks = {6: [4, 4, 4, 4, 2, 2], 5: [4, 4, 4, 4, 2]}


if __name__ == '__main__':

  opts = get_argparser().parse_args()

  n_features = DaLesPointCloudDataset.n_features
  n_classes  = DaLesPointCloudDataset.n_classes


  neighbours =16
  d_out = randla_net_blocks[opts.n_blocks]
  decimations = decimations_blocks[opts.n_blocks]

  opts = get_argparser().parse_args()
  file_path = opts.point_cloud
  model_path = opts.model_path

  if opts.device == 'cuda':
     #os.environ['CUDA_VISIBLE_DEVICES'] =  0 #opts.gpu_id
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  else:
     device = torch.device('cpu')

  dense_cloud = load_open3d_cloud(file_path)
  subsampling = True if opts.subsampling == 'y' else False
  dense_labels = True if opts.dense_labels =='y' else False
  if subsampling:
    sampled_cloud = dense_cloud.uniform_down_sample(opts.step)
    open3d.io.write_point_cloud("sampled_cloud.ply", sampled_cloud)
  else:
    sampled_cloud = dense_cloud

  mean, _ = sampled_cloud.compute_mean_and_covariance()
  coords = np.asarray(sampled_cloud.points) - mean


  coords = coords[np.newaxis,:].astype(np.float32)

  pointCloudDataset = DaLesPointCloudDataset(coords, device=device, n_decimation=decimations, n_neighbors=neighbours)

  checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
  model = RandLANet(n_features, n_classes, d_out)
  model.load_state_dict(checkpoint)
  del checkpoint

  if device.type != 'cpu':
    model.to(device)

  featured_cloud = pointCloudDataset.prep_cloud()

  model.eval()
  with torch.no_grad():
    output = model(featured_cloud)

  predictions = output['predicts'].detach().cpu().numpy().squeeze()

  probs = eval_probs(output['logits'])
  probs = probs.detach().cpu().numpy().squeeze()

  predictions =  threshold_predictions(probs, predictions, threshold=opts.threshold)
  coords = coords.squeeze()

  if dense_labels and subsampling:
    dense_coords = np.asarray(dense_cloud.points) - mean
    predictions =interpolate_dense_labels(coords, predictions, dense_coords, opts.knn_neighbors)
    colored_predictions = DaLesPointCloudDataset.color_decode_target(predictions)
    dense_coords += mean
    write_point_cloud(dense_coords, colored_predictions, opts.output_name)
  else:
    colored_predictions = DaLesPointCloudDataset.color_decode_target(predictions)
    coords +=mean
    write_point_cloud(coords, colored_predictions, opts.output_name)

