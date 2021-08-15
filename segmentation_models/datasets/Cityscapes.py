import os
import torch.utils.data as data
import torchvision.transforms as transforms

from collections import namedtuple
from PIL import Image
import numpy as np

MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET  = [0.229, 0.224, 0.225]

class Cityscapes(data.Dataset):

  CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id',  'color'])
  classes = [
        CityscapesClass('unlabeled',            0,  255,  (0, 0, 0)),
        CityscapesClass('ego vehicle',          1,  255,  (0, 0, 0)),
        CityscapesClass('rectification border', 2,  255,  (0, 0, 0)),
        CityscapesClass('out of roi',           3,  255,  (0, 0, 0)),
        CityscapesClass('static',               4,  255,  (0, 0, 0)),
        CityscapesClass('dynamic',              5,  255,  (111, 74,0)),
        CityscapesClass('ground',               6,  255,  (81, 0, 81)),
        CityscapesClass('road',                 7,  0,    (128, 64, 128)),
        CityscapesClass('sidewalk',             8,  1,    (244, 35, 232)),
        CityscapesClass('parking',              9,  255,  (250, 170, 160)),
        CityscapesClass('rail track',           10, 255,  (230, 150, 140)),
        CityscapesClass('building',             11, 2,    (70, 70, 70)),
        CityscapesClass('wall',                 12, 3,    (102, 102, 156)),
        CityscapesClass('fence',                13, 4,    (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255,  (180, 165, 180)),
        CityscapesClass('bridge',               15, 255,  (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255,  (150, 120, 90)),
        CityscapesClass('pole',                 17, 5,    (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255,  (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6,    (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7,    (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8,    (107, 142, 35)),
        CityscapesClass('terrain',              22, 9,    (152, 251, 152)),
        CityscapesClass('sky',                  23, 10,   (70, 130, 180)),
        CityscapesClass('person',               24, 11,   (220, 20, 60)),
        CityscapesClass('rider',                25, 12,   (255, 0, 0)),
        CityscapesClass('car',                  26, 13,   (0, 0, 142)),
        CityscapesClass('truck',                27, 14,   (0, 0, 70)),
        CityscapesClass('bus',                  28, 15,   (0, 60, 100)),
        CityscapesClass('caravan',              29, 255,  (0, 0, 90)),
        CityscapesClass('trailer',              30, 255,  (0, 0, 110)),
        CityscapesClass('train',                31, 16,   (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17,   (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18,   (119, 11, 32)),
        CityscapesClass('license plate',        -1, 255,  (0, 0, 142)),
    ]

  train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
  train_id_to_color.append([0, 0, 0])
  train_id_to_color = np.array(train_id_to_color)
  id_to_train_id = np.array([c.train_id for c in classes])


  num_classes = 19
  ignore_index = 255
  classes_to_names = {0:'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence', 5: 'pole',
                      6: 'traffic light', 7: 'traffic sign', 8: 'vegetation', 9: 'terrain', 10: 'sky',
                      11: 'person', 12: 'rider', 13: 'car', 14: 'truck', 15: 'bus', 16: 'train',
                      17: 'motorcycle', 18: 'bicycle'}

  def __init__(self, root, split='train', mode='fine', target_type='semantic', image_transform=None, label_transform = None):
    self.root = os.path.expanduser(root)
    self.mode = 'gtFine'
    self.target_type = target_type
    self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
    self.targets_dir = os.path.join(self.root, self.mode, split)
    self.image_transform = image_transform
    self.label_transform = label_transform

    self.split = split
    self.images = []
    self.targets = []

    if split not in ['train', 'test', 'val']:
      raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

    if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
      raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')

    for city in os.listdir(self.images_dir):
      img_dir = os.path.join(self.images_dir, city)
      target_dir = os.path.join(self.targets_dir, city)

    for file_name in os.listdir(img_dir):
      self.images.append(os.path.join(img_dir, file_name))
      target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0], self._get_target_suffix(self.mode, self.target_type))
      self.targets.append(os.path.join(target_dir, target_name))

  @classmethod
  def encode_target(cls, target):
    return cls.id_to_train_id[np.array(target)]

  @classmethod
  def decode_target(cls, target):
    target[target == 255] = 19
    return cls.train_id_to_color[target]

  def __getitem__(self, index):
    image = Image.open(self.images[index]).convert('RGB')
    target = Image.open(self.targets[index])
    if self.image_transform is not None:
       image  = self.image_transform(image)
    if self.label_transform is not None:
       target = self.label_transform(target)
    target = self.encode_target(target)
    return image, target

  def __len__(self):
    return len(self.images)

  def _get_target_suffix(self, mode, target_type):
    if target_type == 'instance':
      return '{}_instanceIds.png'.format(mode)
    elif target_type == 'semantic':
      return '{}_labelIds.png'.format(mode)
    elif target_type == 'color':
      return '{}_color.png'.format(mode)
    elif target_type == 'polygon':
      return '{}_polygons.json'.format(mode)
    elif target_type == 'depth':
      return '{}_disparity.png'.format(mode)


def cityscapes_loaders(root_dir, image_sz=(512, 512), train_batch_sz=4, val_batch_sz=4):

    image_transform = transforms.Compose([transforms.Resize(image_sz, Image.BILINEAR), transforms.ToTensor(),
                                        transforms.Normalize(MEAN_IMAGENET, STD_IMAGENET)])
    targets_transform = transforms.Compose([transforms.Resize(image_sz, Image.NEAREST)])

    train_dataset = Cityscapes(root_dir, split='train', image_transform=image_transform, label_transform=targets_transform)
    val_dataset = Cityscapes(root_dir, split='val', image_transform=image_transform, label_transform=targets_transform)

    print('dataset: {}, train set: {}, validation set: {}'.format('CityScapes', len(train_dataset), len(val_dataset)))

    train_loader = data.DataLoader(train_dataset, train_batch_sz, shuffle=True, num_workers=2)
    val_loader   = data.DataLoader(val_dataset, val_batch_sz, shuffle=True, num_workers=2)
    return train_loader, val_loader,\
           train_dataset.dataset.num_classes, train_dataset.dataset.classes_to_names,\
           Cityscapes.decode_target, train_dataset.dataset.ignore_index