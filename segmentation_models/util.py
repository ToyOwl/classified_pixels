import os
import torch
import argparse

from models import get_segmenation_model
from models import RandLANet

def save_as_jit_model(model, path='jit_model.pth', batch_sz=1, channel_sz=3, img_sz=512):
    model.eval()
    with torch.no_grad():
     in_tensor = torch.randn(batch_sz, channel_sz, img_sz, img_sz, requires_grad=True).to(torch.float32)
     traced_ceil = torch.jit.trace(model, (in_tensor))
     torch.jit.save(traced_ceil, path)
     print('jit model has been successfully saved at path: ', path)

def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_pth', default=None, type=str, help='path to trained model')
    parser.add_argument('--out_model_pth', default=None, type=str, help='path to trained model')
    parser.add_argument('--model', type=str, default='deeplabv3-resnet50', choices=[
                                 'deeplabv3-resnet18',  'deeplabv3-resnet34','deeplabv3-resnet50',
                                 'deeplabv3-resnet101','deeplabv3plus-resnet101', 'deeplabv3-resnet152',
                                 'deeplabv3plus-resnet152'], help='model type')
    parser.add_argument('--n_classes', type=int)
    parser.add_argument('--output_stride', type=int, default=8, choices=[8, 16])
    parser.add_argument('--is_point_cloud', default='false', type=str, choices=['false', 'true'], help='re-converting models for RandLa-Net' )
    parser.add_argument('--finetuned', default='false', type=str, choices=['false', 'true'], help='finetuned RandLa-Net')
    return parser

if __name__ == '__main__':

    opts = get_arg_parser().parse_args()

    checkpoint = torch.load(opts.model_pth, map_location=torch.device('cpu'))

    if isinstance(checkpoint, dict):
       state_dict = checkpoint['model_state']
    else:
       state_dict = checkpoint


    if opts.is_point_cloud == 'false':
        model = get_segmenation_model(model_name=opts.model, num_classes=opts.n_classes, out_stride=opts.output_stride)
    elif opts.is_point_cloud == 'false' and opts.finetuned == 'true':
        state_dict = {k.partition('module.')[2]: state_dict[k] for k in state_dict.keys()}

    out_model_path = opts.out_model_pth
    out_model_name = os.path.basename(out_model_path)
    out_model_location = os.path.dirname(out_model_path)

    if out_model_location:
       if not os.path.exists(out_model_location):
              os.makedirs(out_model_location)

    if opts.is_point_cloud == 'false':
        save_as_jit_model(model, os.path.join(out_model_location, out_model_name))
    else:
        #state_dict = checkpoint['model_state']
        #state_dict = {k.partition('module.')[2]:state_dict[k] for k in state_dict.keys()}
        torch.save(state_dict, os.path.join(out_model_location, out_model_name))

    del checkpoint

