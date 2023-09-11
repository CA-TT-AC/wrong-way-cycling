import argparse
import os
import torch

from angle_prediction import models
from validation_api import validation
from torchvision import transforms
from dataset import NeRFDataset


def get_args_parser():
    parser = argparse.ArgumentParser('Angle prediction', add_help=False)
    parser.add_argument('--resume', default=r'D:\wise_transportation\gitee_repo\mmyolo\ckpt\finetune-checkpoint-90.pth',
                        type=str,
                        help='validation dataset path')
    parser.add_argument('--val_data_path', default=r'D:\Instant-NGP-for-RTX-3000-and-4000\angle_data\val_dataset',
                        type=str,
                        help='validation dataset path')
    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--model', default='AngleModel', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=512, type=int,
                        help='images input size')

    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory '
                             'constraints)')
    return parser


def main(args):
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_val = NeRFDataset(os.path.join(args.val_data_path), transform=transform_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        num_workers=args.num_workers,
        drop_last=True,
        shuffle=False,
    )
    # define the model
    device = torch.device('cuda')
    model = models.__dict__[args.model]()

    model.to(device)
    ckpt = torch.load(args.resume)
    print(ckpt['model'].keys())
    print(model.state_dict().keys())
    model.load_state_dict(ckpt['model'])
    model_without_ddp = model
    val_state = validation(model, data_loader_val,
                           device, 100,
                           log_writer=None,
                           visual=True,
                           args=args)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
