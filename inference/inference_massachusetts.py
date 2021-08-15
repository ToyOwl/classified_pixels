
import torch
import torch.nn as nn
import numpy as np

from   skimage.io import imsave
import albumentations as album
import torchvision.transforms.functional as tf
import cv2
import os

import argparse


MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET  = [0.229, 0.224, 0.225]
TARGET_COLOR  = [255,0,255]

def padding_image(width=1536, height=1536):
    test_transform = [
        album.Resize(height=height, width=width)
    ]
    return album.Compose(test_transform)
    #test_transform = [
    #    album.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0)
    #]
    #return album.Compose(test_transform)


def mask_color_img(img, mask, color=TARGET_COLOR, alpha=0.7):
    mask = mask.squeeze()
    out = img.copy()
    img_layer = img.copy()
    msk  = np.where(mask>0)
    img_layer[msk] = mask[msk]
    out = cv2.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)
    return out


def crop_image(image, target_image_dims=[1500, 1500, 3]):
    target_size = target_image_dims[0]
    image_size = len(image)
    padding = (image_size - target_size) // 2

    return image[padding:image_size - padding, padding:image_size - padding,:]


class TorchScriptInference(object):

  def __init__(self, image_dir, out_dir, model, mask_prefix='mask_', cat_prefix = 'out_'):
    self.model = model
    self.mask_prefix = mask_prefix
    self.cat_prefix  = cat_prefix
    self.padding = padding_image()
    self.out_dir = out_dir
    self.samples = []

    for image_name in os.listdir(image_dir):
      image = cv2.imread(os.path.join(image_dir, image_name))
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image = tf.to_tensor(image)
      self.samples.append(tf.normalize(image, MEAN_IMAGENET, STD_IMAGENET))

  def decode(self):
      self.model.eval()
      with torch.no_grad():
        for idx, image in enumerate(self.samples):
            npimage = image.cpu().numpy().squeeze()
            npimage = npimage.transpose(1,2,0)
            original_size = npimage.shape
            npimage = self.padding(image=npimage)
            npimage = npimage['image'].transpose(2,0,1)
            outputs = self.model(torch.from_numpy(npimage[np.newaxis, : , :, :]))

            out_img = (self.denorm(image) * 255).numpy().astype(np.uint8).squeeze()

            predictions = outputs.detach().max(dim=1)[1].cpu().numpy()

            scaled_size = predictions.shape
            scaled_x   = original_size[1]/scaled_size[1]
            scaled_y   = original_size[0] / scaled_size[0]

            color_predictions = self.decode_predictions(predictions).astype(np.uint8)

            out_img = (self.denorm(image) * 255).numpy().astype(np.uint8).squeeze()
            out_img = out_img.transpose(1, 2, 0)

            masked_img = cv2.resize(color_predictions, (original_size[1], original_size[0]))

            masked_img = mask_color_img(out_img, masked_img)
            imsave(os.path.join(self.out_dir, 'sample_msk.png'), masked_img)

            '''
            out_img = out_img.transpose(1, 2, 0)
            _out_img = np.concatenate((out_img, crop_image(color_predictions)), axis=1)
            mask_file_name = (self.mask_prefix + '_{}.png').format(idx)
            cat_file_name  = (self.cat_prefix + '_{}.png').format(idx)
            imsave(os.path.join(self.out_dir, mask_file_name), _out_img)
            imsave(os.path.join(self.out_dir, cat_file_name), mask_color_img(out_img, crop_image(predictions.transpose(1,2,0))))
            '''

  def denorm(self, tensor, mean=MEAN_IMAGENET, std=STD_IMAGENET):
      mean = np.array(mean)
      std = np.array(std)
      _mean = -mean / std
      _std = 1 / std
      return tf.normalize(tensor, _mean, _std)

  def decode_predictions(self, target, threshold=.0):
    target = target.squeeze()
    out_img = np.zeros((target.shape[0], target.shape[1], 3))
    out_img[np.where(target > threshold)] = [255, 0, 255]
    return out_img

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', default=None, type=str, help='path to  images')
    parser.add_argument('--output_dir', default=None, type=str,  help='path to decoded images')
    parser.add_argument('--model_path', default=None, type=str, help='path to trained torchscipt model')

    parser.add_argument('--mask_prefix', default='mask_', type=str, help='prefix for masked images')
    parser.add_argument('--cat_prefix', default='out_',  type=str, help='prefix for cat images')

    return parser


if __name__ == '__main__':

    opts = get_argparser().parse_args()
    model = torch.jit.load(opts.model_path, map_location=torch.device('cpu'))

    inference = TorchScriptInference(image_dir=opts.image_dir,
                                     out_dir=opts.output_dir,
                                     model=model,
                                     mask_prefix=opts.mask_prefix,
                                     cat_prefix=opts.cat_prefix)
    inference.decode()
