import torch
import numpy as np
import pandas as pd
import os

from PIL import Image
from torchvision import transforms
from scipy.interpolate import griddata

from segmentation_models.datasets.ply import dict2ply, ply2dict

__all__ = ['targets_to_one_hot', 'load_RGB_PIL_image', 'top_view', 'augmentation_transform']

transformer = transforms.Compose([transforms.ToTensor()])
untransformer = transforms.ToPILImage()

class top_view:

   def __init__(self, color_decoder, size):
      self.size = size
      self.color_decoder = color_decoder

   def __call__(self, points, labels):

      if points.shape[1] !=labels.shape[1]:
         raise ValueError('number of batchs do not match')

      grid_x, grid_y = np.mgrid[0:self.size[0], 0:self.size[1]]
      delta = labels.max()+1
      rgb_images =[]
      for idx in range(points.shape[0]):
         points_xy = points[idx, :, :]
         x_points = points_xy[:, 0]
         y_points = points_xy[:, 1]
         x_points = self.size[0] * ((x_points - x_points.min()) / (x_points.max() - x_points.min()))
         y_points = self.size[0] * ((y_points - y_points.min()) / (y_points.max() - y_points.min()))
         img = []
         img.append(x_points)
         img.append(y_points)
         img = np.array(img)
         img = np.transpose(img, (1, 0))
         grid = griddata(np.array(img), (labels[idx,:] + delta).astype(np.float32), (grid_x, grid_y), method='linear', fill_value=np.nan)
         grid = grid - delta
         rgb = self.color_decoder(grid)
         rgb = rgb.transpose(2, 0, 1)
         rgb_images.append(rgb)
      return rgb_images

def targets_to_one_hot(colors, labels, out_type = np.float32):
    if torch.is_tensor(labels):
        labels = torch.from_numpy(np.array(labels))
        labels = torch.squeeze(labels)
        #plt.imshow(labels)
        #plt.show()
        labels = labels.permute(2,0,1).contiguous()
        out = torch.empty(labels.shape[1], labels.shape[2], dtype=torch.long)
        for cls in colors:
          tens = torch.tensor(cls, dtype=torch.uint8).unsqueeze(1).unsqueeze(2)
          idx = (labels == torch.tensor(cls, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
          validx = (idx.sum(0) == 3)
          out[validx] = torch.tensor(colors[cls], dtype=torch.long)
    else:
        #print("Num labels found: %i" % torch.sum(validx) )
       num_classes = len(colors)
       labels = np.array(labels)
       shape = labels.shape[:2]+(len(colors),)
       out = np.zeros(shape, dtype=out_type)
       for idx, cls in enumerate(colors):
          #tout = np.all(np.array(labels).reshape((-1,3)) == cls, axis=1)
          out[:, :, idx] = np.all(np.array(labels).reshape((-1,3)) == cls, axis=1).reshape(shape[:2])
          out = out.transpose(2, 0, 1)

    #print("Num labels found: %i" % torch.sum(out))
    return out

def load_RGB_PIL_image(path, out_type = np.float32, device_type ='cpu'):
    image = Image.open(path).convert('RGB')
    image = transforms(image).unsqueeze(0)
    return image.to(device_type, out_type)

def unload_from_tensor(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = untransformer(image)

names_to_labels = {'glo': 0, 'vegetation': 1, 'building': 2,  'clutter': 3}
names_to_colors = {'glo': (000, 000, 255), 'vegetation': (000,255,000), 'building': (255,000,000), 'clutter': (255, 000,255)}
colors_to_labels ={(000, 000, 255): 0, (000,255,000): 1, (255,000,000): 2, (255, 000,255): 3}
colors_to_names = {(000, 000, 255): 'glo', (000,255,000): 'vegetation', (255,000,000): 'building', (255, 000,255): 'clutter'}

def write_point_cloud(points, colors, path_ply):
    data_ply={
        'x': points[:, 0],
        'y': points[:, 1],
        'z': points[:, 2],
        'red': colors[:, 0],
        'green': colors[:, 1],
        'blue': colors[:, 2]}
    if dict2ply(data_ply, path_ply):
        print('PLY point cloud successfully saved to {}'.format(path_ply))

def csv_to_pointclods(pth_dir_datasets,  subsampling=1, label_decoder=names_to_labels, color_decoder=names_to_colors, verbose_out =True):

    pointclouds,  u_labels = [], []

    for (root, dirs, files) in os.walk(pth_dir_datasets):

        for file in files:

            ext = os.path.split(file)[-1].split('.')[-1]
            file = os.path.split(file)[-1].split('.')[-2]

            if ext == 'csv':

                d_frame = pd.read_csv(os.path.join(pth_dir_datasets, file + '.' + ext),  delimiter=',')

                x = d_frame['x'].to_numpy()
                y = d_frame['y'].to_numpy()
                z = d_frame['z'].to_numpy()

                labels = d_frame['label'].to_numpy(dtype=str)
                labels = np.char.lower(labels)
                uint8_labels = np.array([label_decoder[label] for label in labels]).astype(dtype=np.uint8)
                points = np.vstack((x, y, z))
                points = points.swapaxes(0,1)

                points = points[::subsampling]
                uint8_labels = uint8_labels[::subsampling]

                pointclouds.append(points)
                u_labels.append(uint8_labels)

                color_labels = np.array([color_decoder[label] for label in labels]).astype(dtype=np.uint8)
                color_labels = color_labels[::subsampling]

            if verbose_out:
               write_point_cloud(points, color_labels, file+'.ply')

    return pointclouds,  u_labels

def create_rotations(axis, angle):

    t1 = np.cos(angle)
    t2 = 1 - t1
    t3 = axis[:, 0] * axis[:, 0]
    t6 = t2 * axis[:, 0]
    t7 = t6 * axis[:, 1]
    t8 = np.sin(angle)
    t9 = t8 * axis[:, 2]
    t11 = t6 * axis[:, 2]
    t12 = t8 * axis[:, 1]
    t15 = axis[:, 1] * axis[:, 1]
    t19 = t2 * axis[:, 1] * axis[:, 2]
    t20 = t8 * axis[:, 0]
    t24 = axis[:, 2] * axis[:, 2]

    R = np.stack([t1 + t2 * t3, t7 - t9, t11 + t12,
                  t7 + t9,  t1 + t2 * t15, t19 - t20,
                  t11 - t12, t19 + t20, t1 + t2 * t24], axis=1)
    return np.reshape(R, (-1, 3, 3))

def augmentation_transform(points,scale_min=0.9, scale_max=1.1, rotate_angle=.03,
                           scale_anisotropic=True, augment_symmetries=[False, False, False], augment_noise=0.0025):

   R = np.eye(points.shape[1])

   theta = np.random.rand() * 2 * np.pi
   phi = (np.random.rand() - 0.5) * np.pi

   u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

   alpha = np.random.rand() * rotate_angle * np.pi

   R = create_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]

   R = R.astype(np.float32)

   min_s = scale_min
   max_s = scale_max

   if scale_anisotropic:
      scale = np.random.rand(points.shape[1]) * (max_s - min_s) + min_s
   else:
      scale = np.random.rand() * (max_s - min_s) - min_s

   symmetries = np.array(augment_symmetries).astype(np.int32)
   symmetries *= np.random.randint(2, size=points.shape[1])
   scale = (scale * (1 - symmetries * 2)).astype(np.float32)

   noise = (np.random.randn(points.shape[0], points.shape[1]) * augment_noise).astype(np.float32)

   augmented_points = np.sum(np.expand_dims(points, 2) * R, axis=1) * scale + noise

   return augmented_points