import torch
import torch.nn as nn
import argparse

from utils import FocalLoss, DiceLoss
from utils import PolyLR

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument('--data_root', type=str, default='./datasets/data', help='path to Dataset')

    parser.add_argument('--ckpt_dir', default=None, type=str, help='restore from checkpoint')
    parser.add_argument('--ckpt', default=None, type=str, help='restore from checkpoint')

    # Deeplab Options
    parser.add_argument('--model', type=str, default='deeplabv3-resnet50',
                        choices=['deeplabv3-resnet18',  'deeplabv3-resnet34',
                                 'deeplabv3-resnet50',  'deeplabv3-resnet101',
                                 'deeplabv3plus-resnet101', 'deeplabv3-resnet152',
                                 'deeplabv3plus-resnet152'], help='model type')

    parser.add_argument('--output_stride', type=int, default=16, choices=[8, 16])

    parser.add_argument('--points', type=int, default=65536, help='crop sub point cloud size (default: 65536)')
    parser.add_argument('--neighbours', type=int, default=16, help='number of neighbours (default: 16)')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size (default: 4)')
    parser.add_argument('--val_batch_size', type=int, default=4, help='batch size for validation (default: 4)')
    parser.add_argument('--crop_size', type=int, default=712)

    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--gpu_id', type=str, default='0',help='GPU ID')


    parser.add_argument('--print_interval', type=int, default=5, help='print interval of loss (default: 10)')


    parser.add_argument('--iterations', type=int, default=30e3, help='maximum interations number (default: 30k)')
    parser.add_argument('--epochs', type=int, default=300, help='maximum interations number (default: 30k)')
    parser.add_argument('--lr_drop_period', type=int, default=10000, help='poly scheduler drop period (default: 10k)')
    parser.add_argument('--lr_drop_factor', type=float, default=.65)
    parser.add_argument('--l2_regularization', type=float, default=1e-10,
                        help='regularization parameter adam, rmsprop (default: 1e-10)')
    parser.add_argument('--lr_initial', type=float, default=0.01, help='initial learning rate (default: 0.01)')
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight decay (default: 1e-4)')

    parser.add_argument('--momentum',   type=float, default=0.9, help='SGD, RMSProp momentum (default: 0.9)')
    parser.add_argument('--loss_type', type=str, default='focal_loss',
                        choices=['cross_entropy', 'focal_loss', 'dice_loss'], help="loss type (default: focal)")
    parser.add_argument('--lr_scheduler', type=str, default='poly', choices=['poly', 'step', 'lambda', 'exp', 'cosine_annealing_warm_restarts'],
                        help='learning rate scheduler  (default: poly)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['rprop', 'adam', 'rmsprop', 'sgd'],
                        help='loss function optimizer  (default: adam)')



    parser.add_argument('--vis_port', type=str, default='8097',help='port for visdom')
    parser.add_argument('--vis_num_samples', type=int, default=4, help='number of samples for visualization (default: 8)')
    return parser

def config_optimizer_sheduler(model, opt, has_encoder=True):

  if has_encoder:
    params = [{'params': model.backbone.parameters(), 'lr':  0.1 *opt.lr_initial},
              {'params': model.classifier.parameters(), 'lr': opt.lr_initial}]
  else:
    params = [{'params': model.parameters(), 'lr':  opt.lr_initial}]

  def_optimizer = torch.optim.SGD(params, lr=opt.lr_initial, weight_decay=opt.weight_decay, momentum=opt.momentum)

  optimizers = {'sgd': def_optimizer,
                'rprop': torch.optim.Rprop(params, lr=opt.lr_initial),
                'adam': torch.optim.Adam(params, lr=opt.lr_initial,  weight_decay=opt.weight_decay, eps=opt.l2_regularization),
                'rmsprop': torch.optim.RMSprop(params, lr=opt.lr_initial, weight_decay=opt.weight_decay,
                                                momentum=opt.momentum, eps=opt.l2_regularization)}

  optimizer = optimizers.get(opt.optimizer, def_optimizer)
  default_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop_period)
  factor = opt.lr_drop_factor
  lr_factor = lambda epoch: factor ** epoch
  scheduler_map = {'step': default_scheduler,
                   'poly': PolyLR(optimizer,opt.iterations),
                   'lambda': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor),
                   'exp': torch.optim.lr_scheduler.ExponentialLR(optimizer, opt.lr_drop_factor),
                   'cosine_annealing_warm_restarts':torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=5e-5)}

  return optimizer, scheduler_map.get(opt.lr_scheduler, default_scheduler)

def config_loss_function(loss_type='cross_entropy', ignore_index=255):
  if loss_type == 'focal_loss':
     criterion = FocalLoss(ignore_index, size_average=True)
  elif loss_type == 'cross_entropy':
     criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
  elif loss_type == 'dice_loss':
     criterion = DiceLoss()
  return criterion