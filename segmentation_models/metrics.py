import torch
import numpy as np

class SegmentationMetrics(object):

  eps = 1e-12

  def __init__(self, n_classes, normalized=False, ignore_index=None):

    self.n_classes = n_classes
    self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
    self.normalized = normalized
    self.ignore_index = ignore_index

  def reset(self):
    self.confusion_matrix.fill(0)

  #predictions or targets must be (N,H,W) ot (N,K,H,W) tensors N - examples number, K - classes number
  def add_measurements(self, predictions, targets):

    if isinstance(predictions, torch.Tensor) and not isinstance(predictions, np.ndarray):
      raise ValueError('predictions must be tensors or numpy ndarrays')

    if isinstance(targets, torch.Tensor) and not isinstance(targets, np.ndarray):
      raise ValueError('targets must be tensors or numpy ndarrays')

    if torch.is_tensor(predictions):
      predictions = predictions.cpu().numpy()

    if torch.is_tensor(targets):
      targets = targets.cpu.numpy()

    if predictions.shape[0] != targets.shape[0]:
      raise ValueError('number of targets and predicted outputs do not match')

    if np.ndim(predictions) == 4:

      if predictions.shape[1] != self.n_classes:
        raise ValueError('number of predictions does not match size of confusion matrix')

      predictions = np.argmax(predictions, 1)

    else:
      if (predictions.max() >= self.n_classes) and (predictions.min() < 0):
        raise ValueError('predicted values are not between 0 and k-1')

    for prediction, target in zip(predictions, targets):
        self._eval_conf_matrix(prediction.flatten(), target.flatten())

  def eval(self):

    confusion_matrix = self.confusion_matrix

    if self.normalized:
        confusion_matrix = self._normalize_conf_matrix(confusion_matrix)

    if self.ignore_index is not None:
       confusion_matrix[:, self.ignore_index] = 0
       confusion_matrix[self.ignore_index, :] = 0

    tp = np.diag(confusion_matrix)
    fp = confusion_matrix.sum(0) - tp
    fn = confusion_matrix.sum(1) - tp

    with np.errstate(divide='ignore', invalid='ignore'):
      iou = tp/(tp + fp + fn)
      miou = np.nanmean(iou)
      acc = tp.sum() / confusion_matrix.sum()
      macc = np.nanmean(tp/confusion_matrix.sum(1))
      freq = confusion_matrix.sum(axis=1)/confusion_matrix.sum()
      fwacc = (freq[freq > 0] * iou[freq > 0]).sum()
      cls_iou = dict(zip(range(self.n_classes), iou))

      return {'IoU': cls_iou, 'mIoU' : miou, 'ACC': acc, 'mACC': macc, 'fwACC': fwacc}

  def _eval_conf_matrix(self, predictions, targets):

      mask = (targets >= 0) & (targets < self.n_classes)

      confusion_matrix = np.bincount((targets[mask]+ self.n_classes * predictions[mask]).astype(int),
                                     minlength=self.n_classes ** 2)
      confusion_matrix = confusion_matrix.reshape((self.n_classes, self.n_classes))
      self.confusion_matrix += confusion_matrix

  def _normalize_conf_matrix(self, confusion_matrix):
      if self.normalized:
        confusion_matrix = confusion_matrix.astype(np.float32)
        return confusion_matrix/confusion_matrix.sum(1).clip(min=SegmentationMetrics.eps)[:,None]
      else:
        return confusion_matrix

def iou_metric(y_pred, y_true):

    hist = np.histogram2d(y_true.flatten(), y_pred.flatten(), bins=([0, 0.5, 1], [0, 0.5, 1]))

    intersection = hist[0]

    area_true = np.histogram(y_true, bins=[0, 0.5, 1])[0]
    area_pred = np.histogram(y_pred, bins=[0, 0.5, 1])[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    union = area_true + area_pred - intersection

    intersection = intersection[1:, 1:]
    intersection[intersection == 0] = 1e-9

    union = union[1:, 1:]
    union[union == 0] = 1e-9

    iou = intersection / union
    return np.mean(iou)
