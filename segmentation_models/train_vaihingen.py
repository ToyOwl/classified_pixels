from config import get_argparser
from datasets import get_vaihingen_dataloaders
from train import PointCloudTrain

import torch

if __name__ == '__main__':

    torch.manual_seed(0)

    num_classes=4
    num_features=3
    d_out = [16, 64, 128, 256, 512, 1024]
    decimations = [4, 4, 4, 4, 2, 2]

    opts = get_argparser().parse_args()

    trainer = PointCloudTrain(get_vaihingen_dataloaders, opts, decimations=decimations, d_outs=d_out)
    trainer.run()