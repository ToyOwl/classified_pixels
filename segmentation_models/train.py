import torch
import os
import torch.nn as nn
import numpy as np

from utils import VisdomVisualizer, Denormalize
from metrics import *

from config import config_optimizer_sheduler, config_loss_function
from models import get_segmenation_model
from models import RandLANet


class Train(object):

  def __init__(self, dataloader, train_opts):

      if train_opts.device == 'cuda':
         os.environ['CUDA_VISIBLE_DEVICES'] = train_opts.gpu_id
         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      else:
         self.device = torch.device('cpu')

      self.model_name = train_opts.model
      self.print_interval = train_opts.print_interval
      self.max_iterations = train_opts.iterations
      self.max_epochs = train_opts.epochs
      self.ckpt_dir = train_opts.ckpt_dir
      self.ckpt = train_opts.ckpt
      self.opts = train_opts
      self.visualizer = VisdomVisualizer(port=train_opts.vis_port)
      self.train_len, self.validate_len = 0, 0

  def run(self):

     self.visualizer.plot_table('options', vars(self.opts))

     print("#==========   Train Loop   ==========#")

     if self.device.type != 'cpu':
        self.model = nn.DataParallel(self.model)

     self.model.to(self.device)

     self.best_score = 0.0
     self.current_iter = 0.0
     self.cur_epochs = 0
     self.interval_loss = 0

     while True:
       self.model.train()
       self.run_impl()
       self.cur_epochs = self.cur_epochs +1
       if self.current_iter >= self.max_iterations or self.cur_epochs > self.max_epochs:
         self.save_ckpt(self.ckpt_dir + '/max_iter_{}.pth'.format(self.model_name))
         return

  def run_impl(self):
      pass

  def validate(self, full=True):
      for batch in self.val_loader:
          out_val = self.validate_impl(batch)
          if not full:
            break
      if not full:
          return out_val


  def validate_impl(self, batch):
      pass

  def save_ckpt(self, path):

   if self.device.type == 'cpu':
      model_state = self.model.state_dict()
   else:
      model_state = self.model.module.state_dict()

   torch.save({'cur_iter': self.current_iter, 'model_state':  model_state,
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict(),
                'best_score': self.best_score}, path)
   print('model saved as {}'.format(path))

  def save_model_state(self, path):
      if self.device.type == 'cpu':
          model_state = self.model.state_dict()
      else:
          model_state = self.model.module.state_dict()

      torch.save(model_state, path)
      print('model state saved as {}'.format(path))


  def load_model_state(self, path):

    checkpoint = torch.load(path, map_location=torch.device('cpu'))

    if isinstance(checkpoint, dict):
        self.model.load_state_dict(checkpoint['model_state'])
    else:
        self.model.load_state_dict(checkpoint)

    if self.device.type != 'cpu':
       self.model = nn.DataParallel(self.model)
       self.model.to(self.device)


    print('Model restored from {}'.format(path))
    del checkpoint

  def load_ckpt(self, path):
   checkpoint = torch.load(path, map_location=torch.device('cpu'))
   self.model.load_state_dict(checkpoint['model_state'])

   if self.device.type != 'cpu':
      self.model = nn.DataParallel(self.model)
      self.model.to(self.device)

   self.optimizer.load_state_dict(checkpoint['optimizer_state'])
   self.scheduler.load_state_dict(checkpoint['scheduler_state'])
   self.best_score = checkpoint['best_score']
   self.curr_iter = checkpoint['cur_iter']

   print('Training state restored from {}'.format(path))
   print('Model restored from {}'.format(path))
   del checkpoint

  def write_metics(self, file_name, metrics, class_to_name):
      score_str = ''
      for k, v in metrics.items():
          if k != 'IoU':
              score_str += '{}: {}\n'.format(k, v)
          else:
              for key in metrics['IoU']:
                  score_str += '{}: {}\n'.format(class_to_name[key], metrics['IoU'][key])

      with open(os.path.join(self.ckpt_dir, file_name), 'w') as metric_file:
          metric_file.write(score_str)

  def write_iou(self, file_name, iou, class_to_name):
      score_str = 'IoU {}:, {}'.format(iou, class_to_name[0])
      with open(file_name, 'w') as metric_file:
          metric_file.write(score_str)


class ImageTrain(Train):

  def __init__(self, dataloader, train_opts):

     Train.__init__(self,  dataloader, train_opts)

     self.train_loader, self.val_loader, self.n_classes, self.class_to_names, self.decoder, self.ignore_index=\
         dataloader(train_opts.data_root, train_opts.data_root, image_sz=train_opts.crop_size,
                    train_batch_sz=train_opts.batch_size, val_batch_sz=train_opts.val_batch_size)

     self.out_stride = train_opts.output_stride
     self.criterion = config_loss_function(train_opts.loss_type,  self.ignore_index)
     self.iou = SegmentationMetrics(self.n_classes)
     self.model = get_segmenation_model(train_opts.model, self.n_classes, out_stride=self.out_stride)
     self.optimizer, self.scheduler = config_optimizer_sheduler(self.model, train_opts)
     self.denorm = Denormalize()
     self.train_len = len(self.train_loader)
     self.validate_len = len(self.val_loader)

  def run_impl(self):

    train_epoch_loss = 0
    validate_epoch_loss = 0
    train_epoch_iou = 0
    validate_epoch_iou = 0

    for (images, labels) in self.train_loader:

        images = images.to(self.device, dtype=torch.float32)
        labels = labels.to(self.device, dtype=torch.long)

        self.optimizer.zero_grad()

        outputs = self.model(images)
        labels = labels.to(self.device)

        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step(self.cur_epochs + self.current_iter / self.train_len)

        targets = labels.detach().cpu().numpy()
        if self.n_classes > 1:
           predictions= outputs.detach().max(dim=1)[1].cpu().numpy()
           self.iou.reset()
           self.iou.add_measurements(predictions, targets)
           score = self.iou.eval()
           self.write_metics('metrics.txt', score, self.class_to_names)
           score = score['mIoU']
        else:
           predictions = outputs.cpu().numpy()
           score = iou_metric(predictions, targets)
           self.write_metics('metrics.txt', score, self.class_to_names)

        np_loss = loss.detach().cpu().numpy()

        train_epoch_loss += np_loss
        train_epoch_iou += score
        self.interval_loss += np_loss

        self.visualizer.plot_func(self.current_iter / self.train_len, np_loss, 'Loss', 'train loss')

        val_score, ret_samples, val_loss = self.validate(False)
        validate_epoch_loss +=val_loss
        validate_epoch_iou +=val_score

        if (self.current_iter) % self.print_interval == 0:
            self.interval_loss = self.interval_loss / self.print_interval
            print('Epoch {}/{}, Itrs {}/{}, Loss={}, Score={} '.format(
                self.cur_epochs, self.max_epochs, self.current_iter, self.max_iterations, self.interval_loss, val_score))
            self.save_ckpt(self.ckpt_dir + '/last_{}.pth'.format(self.model_name))
            self.interval_loss = 0.0

        if val_score > self.best_score:
           self.best_score = val_score
           self.save_ckpt(self.ckpt_dir + '/best_{}.pth'.format(self.model_name))

        self.visualizer.plot_func(self.current_iter / self.train_len, val_loss, 'Loss', 'validate loss')
        self.visualizer.plot_func(self.current_iter / self.train_len, val_score, 'IoU', 'validate mIoU')
        self.visualizer.plot_func(self.current_iter / self.train_len, score, 'IoU', 'train mIoU')

        for k, (img, target, lbl) in enumerate(ret_samples):
            plot_img = np.concatenate((img, lbl, target), axis=1).astype(np.uint8).transpose(2,0,1)
            self.visualizer.plot_image('sample {}'.format(k), plot_img)

        self.current_iter += 1
        self.model.train()


    self.visualizer.plot_func(self.cur_epochs, validate_epoch_loss / self.train_len, 'epoch Loss', 'validate Loss')
    self.visualizer.plot_func(self.cur_epochs, train_epoch_loss / self.train_len, 'epoch Loss', 'train Loss')
    self.visualizer.plot_func(self.cur_epochs, validate_epoch_iou / self.train_len, 'epoch IoU', 'validate mIoU')
    self.visualizer.plot_func(self.cur_epochs, train_epoch_iou / self.train_len, 'epoch IoU', 'train mIoU')

    if (self.cur_epochs + 1) % self.print_interval ==0:
        self.save_ckpt(self.ckpt_dir + '/epoch_{}_{}.pth'.format(self.model_name, self.cur_epochs+1))

  def validate_impl(self, batch):
      self.model.eval()
      with torch.no_grad():
        images, labels = batch[0], batch[1]
        images = images.to(self.device, dtype=torch.float32)
        labels = labels.to(self.device, dtype=torch.long)
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        targets = labels.cpu().numpy()

        if self.n_classes > 1:
           self.iou.reset()
           predictions= outputs.detach().max(dim=1)[1].cpu().numpy()
           self.iou.add_measurements(predictions, targets)
           score = self.iou.eval()
           self.write_metics('metrics.txt', score, self.class_to_names)
           score = score['mIoU']
        else:
           predictions = outputs.cpu().numpy()
           score = iou_metric(predictions, targets)
           self.write_metics('metrics.txt', score, self.class_to_names)

        ret_samples = []
        for idx in range(images.shape[0]):
          out_img = images[idx].detach().cpu()
          target = targets[idx]
          predict = predictions[idx]
          out_img = (self.denorm(out_img) * 255).numpy().astype(np.uint8).transpose(1,2,0)
          predict = self.decoder(predict[:,:,np.newaxis]).astype(np.uint8)
          target = self.decoder(target[:,:,np.newaxis]).astype(np.uint8)
          ret_samples.append((out_img,predict,target))

        self.model.train()
        return score, ret_samples, loss.detach().cpu().numpy()


class PointCloudTrain(Train):

  def __init__(self,  dataloader, train_opts, decimations, d_outs):

    Train.__init__(self, dataloader, train_opts)
    self.train_loader, self.val_loader, self.n_features, self.n_classes, self.class_to_names, self.top_view_generator = \
        dataloader(train_opts.train_dir, train_opts.val_dir, train_opts.device,
                   train_opts.batch_size, train_opts.points, decimations, train_opts.neighbours)

    self.iou = SegmentationMetrics(self.n_classes)
    self.model = RandLANet(self.n_features, self.n_classes, d_outs)

    self.optimizer, self.scheduler = config_optimizer_sheduler(self.model, train_opts, has_encoder=False)

    self.train_len = len(self.train_loader)
    self.validate_len = len(self.val_loader)

    self.criterion = config_loss_function()

  def run_impl(self):
    train_epoch_loss = 0
    validate_epoch_loss = 0
    train_epoch_iou = 0
    validate_epoch_iou = 0

    for batch in self.train_loader:
        self.current_iter += 1
        self.optimizer.zero_grad()
        outputs = self.model(batch)
        loss = self.criterion(outputs['logits'], outputs['labels'])
        loss.backward()

        self.optimizer.step()
        self.scheduler.step(self.cur_epochs + self.current_iter / self.train_len)

        self.iou.reset()

        predictions = outputs['predicts'].detach().cpu().numpy()
        targets = outputs['labels'].detach().cpu().numpy()

        self.iou.add_measurements(predictions.transpose(0, 1), targets.transpose(0, 1))
        score = self.iou.eval()

        np_loss = loss.detach().cpu().numpy()
        train_epoch_loss += np_loss
        train_epoch_iou += score['mIoU']

        self.interval_loss += np_loss

        self.visualizer.plot_func(self.current_iter / self.train_len, np_loss, 'Loss', 'train loss')
        self.visualizer.plot_func(self.current_iter / self.train_len, score['mIoU'], 'IoU', 'train mIoU')

        val_score, ret_samples, val_loss = self.validate(False)

        self.write_metics('metrics.txt', val_score, self.class_to_names)

        validate_epoch_loss += val_loss
        validate_epoch_iou += val_score['mIoU']

        if self.current_iter % self.print_interval == 0:
           self.interval_loss =  self.interval_loss / self.print_interval
           print('Epoch {}/{}, Itrs {}/{}, Loss={}, Score={} '.format(
               self.cur_epochs, self.max_epochs, self.current_iter, self.max_iterations, self.interval_loss, val_score['mIoU']))
           self.save_ckpt(self.ckpt_dir + '/last_{}.pth'.format(self.model_name))
           self.interval_loss = 0.0

        if val_score['mIoU'] > self.best_score:
           self.best_score = val_score['mIoU']
           self.save_ckpt(self.ckpt_dir + '/best_{}.pth'.format(self.model_name))

        self.visualizer.plot_func(self.current_iter / self.train_len, val_loss, 'Loss', 'validate loss')
        self.visualizer.plot_func(self.current_iter / self.train_len, val_score['mIoU'], 'IoU', 'validate mIoU')

        for k, (target, lbl) in enumerate(ret_samples):
            plot_img = np.concatenate((target, lbl), axis=2)
            self.visualizer.plot_image('sample {}'.format(k), plot_img)

    self.visualizer.plot_func(self.cur_epochs, validate_epoch_loss / self.train_len, 'epoch Loss', 'validate loss')
    self.visualizer.plot_func(self.cur_epochs, train_epoch_loss / self.train_len, 'epoch Loss', 'train loss')
    self.visualizer.plot_func(self.cur_epochs, validate_epoch_iou / self.train_len, 'epoch IoU', 'validate mIoU')
    self.visualizer.plot_func(self.cur_epochs, train_epoch_iou / self.train_len, 'epoch IoU', 'train mIoU')

    if (self.cur_epochs + 1) % self.print_interval == 0:
       self.save_ckpt(self.ckpt_dir + '/epoch_{}_{}.pth'.format(self.model_name, self.cur_epochs + 1))

  def validate_impl(self, batch):
    self.iou.reset()
    ret_samples = []
    with torch.no_grad():
      outputs = self.model(batch)
      loss = self.criterion(outputs['logits'], outputs['labels'])
      predictions = outputs['predicts'].detach().cpu().numpy()
      targets = outputs['labels'].detach().cpu().numpy()
      self.iou.add_measurements(predictions.transpose(0, 1), targets.transpose(0, 1))
      score = self.iou.eval()
      for idx in range(batch['features'].shape[0]):
        points = batch['features'].detach().cpu().transpose(-2, -1).numpy()
        labels = batch['labels'].detach().cpu().numpy()
        predictions = outputs['predicts'].detach().cpu().numpy()
        points = points[idx, :, :]
        points = points[np.newaxis, :, :]
        labels = labels[idx, :]
        labels = labels[np.newaxis, :]
        predictions = predictions[idx, :]
        predictions = predictions[np.newaxis, :, np.newaxis]
        ret_samples.append(
                    (self.top_view_generator(points, labels)[0], self.top_view_generator(points, predictions)[0]))

    self.model.train()
    return score, ret_samples, loss.detach().cpu()