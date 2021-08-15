import torch.nn as nn
import torch.nn.functional as F
import torch 

class FocalLoss(nn.Module):
  def __init__(self, ignore_index, alpha=0.8, gamma=2.0, size_average=True):
    super(FocalLoss, self).__init__()
    self.ignore_index = ignore_index
    self.alpha = alpha
    self.gamma = gamma
    self.size_average = size_average

  def forward(self, inputs, targets):
    ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
    ce_exp = torch.exp(-ce_loss)
    focal_loss = self.alpha * (1-ce_exp)**self.gamma * ce_loss
    if self.size_average:
      return focal_loss.mean()
    else:
      return focal_loss.sum()

class DiceLoss(nn.Module):

   def __init__(self, eps=1e-7):
       super(DiceLoss, self).__init__()
       self.eps = eps

   def forward(self, predictions, target):
      if not torch.is_tensor(predictions):
         raise TypeError('Input type is not a torch.Tensor got {}'.format(type(predictions)))

      if not len(predictions.shape) == 4:
         raise ValueError('Invalid input shape, we expect BxNxHxW got: {}'.format(predictions.shape))

      if not predictions.shape[-2:] == target.shape[-2:]:
          raise ValueError('Input and target shapes must be the same got: {}'.format(predictions.shape, predictions.shape))

      if not predictions.device == target.device:
          raise ValueError('Input and target must be in the same device got: {}'.format(predictions.device, target.device))

      n_classes = predictions.shape[1]

      if n_classes == 1:
         target_one_hot = torch.eye(n_classes + 1)[target.squeeze(1)]
         target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
         true_1_hot_f = target_one_hot[:, 0:1, :, :]
         true_1_hot_s = target_one_hot[:, 1:2, :, :]
         target_one_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
         pos_prob = torch.sigmoid(predictions)
         neg_prob = 1 - pos_prob
         input_soft = torch.cat([pos_prob, neg_prob], dim=1)
      else:
         target_one_hot = torch.eye(n_classes)[target.squeeze(1)]
         target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
         input_soft = F.softmax(predictions, dim=1)

      target_one_hot = target_one_hot.type(predictions.type())
      dims = (0,) + tuple(range(2, predictions.ndimension()))
      intersection = torch.sum(input_soft * target_one_hot, dims)
      cardinality = torch.sum(input_soft + target_one_hot, dims)
      dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
      return (1 - dice_loss)