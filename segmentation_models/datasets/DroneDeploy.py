import os
import numpy as np
import torch.utils.data as data

import torch
from torchvision import transforms
from collections import namedtuple

from PIL import Image
from .utils import *

__all__ = ['DroneDeploy', 'drone_deploy_loaders', 'MEAN_IMAGENET','STD_IMAGENET']

MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET  = [0.229, 0.224, 0.225]

class DroneDeploy(data.Dataset):
  """ DroneDeploy dataset
  """
  DroneDeployClass = namedtuple('DroneDeployClass', ['name', 'id', 'train_id', 'color'])

  classes = [
        DroneDeployClass('building',   0, 0,     (230, 25,   75)),
        DroneDeployClass('clutter',    1, 255,   (145, 30,  180)),
        DroneDeployClass('vegetation', 2, 1,     (60, 180,  75)),
        DroneDeployClass('water',      3, 2,     (245, 130, 48)),
        DroneDeployClass('ground',     4, 3,     (255, 255, 255)),
        DroneDeployClass('car',        5, 4,     (0, 130, 200)),
        DroneDeployClass('ignore',     -1, 255,  (255, 000, 255))]

  num_classes = 5
  ignore_index = 255
  classes_to_names = {0: 'building',  1: 'vegetation', 2: 'water', 3: 'gound', 4: 'car'}

  train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
  train_id_to_color.append([0, 0, 0])
  train_id_to_color = np.array(train_id_to_color)
  id_to_train_id = np.array([c.train_id for c in classes])

  def __init__(self, root_dir, image_transforms=None, target_transforms=None):
   self.label_dir = root_dir + '/labels'
   self.images_dir = root_dir + '/images'
   self.image_transforms = image_transforms
   self.targets_transforms = target_transforms

   if not os.path.isdir(self.images_dir) or not os.path.isdir(self.label_dir):
     raise RuntimeError('Dataset not found or incomplete')

   mask_names = os.listdir(self.label_dir + '/')
   image_names = os.listdir(self.images_dir + '/')
   self.samples = [(os.path.join(self.images_dir, images), os.path.join(self.label_dir, maskes)) for (images, maskes) in
                   zip(image_names, mask_names)]

  @classmethod
  def encode_target(cls, target):
    labels = np.array(target)
    out = np.zeros((labels.shape[0], labels.shape[1]), dtype=np.long)
    for idx, color in enumerate(cls.classes):
      mask = np.where(np.all(labels == color.color, axis=-1))
      out[mask] = color.id
    out =cls.id_to_train_id[np.array(out)]
    return torch.from_numpy(out)

  @classmethod
  def decode_target(cls, target):
    target[target == 255] = 5
    return cls.train_id_to_color[target]

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    image_name, label_name = self.samples[idx]
    image = Image.open(image_name).convert('RGB')
    target  = Image.open(label_name).convert('RGB')

    if self.image_transforms is not None:
      image = self.image_transforms(image)

    if self.targets_transforms is not None:
      target = self.targets_transforms(target)
    target = self.encode_target(target)
    return image, target


def drone_deploy_loaders(train_root_dir, validate_root_dir, image_sz=512, train_batch_sz=4, val_batch_sz=4, train_sz =.7):

    image_transform = transforms.Compose([transforms.Resize((image_sz,image_sz), Image.BILINEAR), transforms.ToTensor(),
                                        transforms.Normalize(MEAN_IMAGENET, STD_IMAGENET)])
    targets_transform = transforms.Compose([transforms.Resize((image_sz,image_sz), Image.NEAREST)])

    dataset = DroneDeploy(train_root_dir, image_transform, targets_transform)
    sz = int(len(dataset) * train_sz )
    train_dataset, val_dataset = data.random_split(dataset, [sz, len(dataset) -sz])

    print('dataset: {}, train set: {}, validation set: {}'.format('DroneDeploy', len(train_dataset), len(val_dataset)))

    train_loader = data.DataLoader(train_dataset, train_batch_sz, shuffle=True, num_workers=2)
    val_loader   = data.DataLoader(val_dataset, val_batch_sz, shuffle=True, num_workers=2)
    return train_loader, val_loader, train_dataset.dataset.num_classes, train_dataset.dataset.classes_to_names, \
           DroneDeploy.decode_target, train_dataset.dataset.ignore_index