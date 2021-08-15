import argparse

from train import PointCloudTrain
from datasets import get_dales_dataloaders

def get_argparser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default=None, type=str, help='path to train data')
    parser.add_argument('--val_dir', default=None, type=str, help='path to validate data')
    parser.add_argument('--ckpt_dir', default=None, type=str, help='checkpoints dir')
    parser.add_argument('--ckpt', default=None, type=str, help='restore from checkpoint')

    parser.add_argument('--lr_drop_period', type=int, default=10000, help='poly scheduler drop period (default: 10k)')
    parser.add_argument('--lr_drop_factor', type=float, default=.65)
    parser.add_argument('--l2_regularization', type=float, default=1e-10,
                        help='regularization parameter adam, rmsprop (default: 1e-10)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD, RMSProp momentum (default: 0.9)')

    parser.add_argument('--lr_initial', type=float, default=0.01, help='initial learning rate (default: 0.01)')
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight decay (default: 1e-4)')

    parser.add_argument('--lr_scheduler', type=str, default='poly', choices=['poly', 'step', 'lambda', 'exp',
                                                                             'cosine_annealing_warm_restarts'],
                        help='learning rate scheduler  (default: poly)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['rprop', 'adam', 'rmsprop', 'sgd'],
                        help='loss function optimizer  (default: adam)')

    parser.add_argument('--points', type=int, default=65536, help='crop sub point cloud size (default: 65536)')
    parser.add_argument('--neighbours', type=int, default=16, help='number of neighbours (default: 16)')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size (default: 4)')
    parser.add_argument('--val_batch_size', type=int, default=4, help='batch size for validation (default: 4)')

    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--gpu_id', type=str, default='0',help='GPU ID')

    parser.add_argument('--print_interval', type=int, default=5, help='print interval of loss (default: 10)')
    parser.add_argument('--val_interval', type=int, default=100,  help='epoch interval for eval (default: 100)')
    parser.add_argument('--iterations', type=int, default=3000e3, help='maximum interations number (default: 30k)')
    parser.add_argument('--epochs', type=int, default=300, help='maximum interations number (default: 30k)')

    parser.add_argument('--n_blocks',    default=5, type=int, choices=[6, 5], help='RandLaNet blocks number')

    parser.add_argument('--vis_port', type=str, default='8097',help='port for visdom')

    parser.add_argument('--model', type=str, default='randlanet', help='NN model')

    return parser

randla_net_blocks ={6: [16, 64, 128, 256, 512, 1024], 5: [16, 64, 128, 256, 512]}
decimations_blocks = {6: [4, 4, 4, 4, 2, 2], 5: [4, 4, 4, 4, 2]}

if __name__ == '__main__':

   opts = get_argparser().parse_args()

   d_out = randla_net_blocks[opts.n_blocks]
   decimations = decimations_blocks[opts.n_blocks]

   trainer = PointCloudTrain(get_dales_dataloaders, opts, decimations=decimations, d_outs=d_out)

   if opts.ckpt:
       trainer.load_model_state(opts.ckpt)

   trainer.run()