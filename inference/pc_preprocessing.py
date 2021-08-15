import os
import open3d
import torch
import numpy as np

from segmentation_models.datasets.pts import read_pts

def threshold_predictions(logits, predictions, threshold =.1, ignore_index=255):
    out_predictions = np.full(predictions.shape, fill_value=ignore_index, dtype=np.long)
    msk = np.where(logits> threshold)
    out_predictions[msk] = predictions[msk]
    return out_predictions

def interpolate_dense_labels(sparse_points, sparse_labels, dense_points, k=3):
    sparse_pcd = open3d.geometry.PointCloud()
    sparse_pcd.points = open3d.utility.Vector3dVector(sparse_points)
    sparse_pcd_tree = open3d.geometry.KDTreeFlann(sparse_pcd)

    dense_labels = []
    for dense_point in dense_points:
        _, sparse_indexes, _ = sparse_pcd_tree.search_knn_vector_3d(dense_point, k)
        knn_sparse_labels = sparse_labels[sparse_indexes]
        dense_label = np.bincount(knn_sparse_labels).argmax()
        dense_labels.append(dense_label)

    return np.asarray(dense_labels)

def eval_probs(logits):
   logits = torch.nn.functional.softmax(logits, dim=-2)
   probs, _ = torch.max(logits, dim=-2)
   return probs

def load_open3d_cloud(file_path):

    ext = os.path.split(file_path)[-1].split('.')[-1]
    if ext == 'pts':
        is_pts = True
    elif ext == 'ply':
        is_pts = False
    else:
        raise RuntimeError('Unknown pointcloud file format')

    if is_pts:
        data_pts = read_pts(file_path)
        pcd = open3d.geometry.PointCloud()
        data_pts = data_pts[:, :3]
        pcd.points = open3d.utility.Vector3dVector(data_pts)
        return pcd
    else:
       return open3d.io.read_point_cloud(file_path)