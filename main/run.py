import os
import argparse
import torch
import __init_path
import shutil

from funcs_utils import save_checkpoint, save_plot, check_data_pararell, count_parameters
from core.config import cfg, update_config

parser = argparse.ArgumentParser(description='Estimate RULA and REBA score')
parser.add_argument('--gpu', type=str, default='0', help='assign multi-gpus by comma concat')
parser.add_argument('--type', type=str, default='REBA', help='Score type')
parser.add_argument('--input', type=str, default='example/input.mp4', help='input video')
parser.add_argument('--info', type=str, default='example/additional_information.json', help='input additional_information.json')
parser.add_argument('--output', type=str, default='output', help='output directory')
parser.add_argument('--visualize', type=bool, default=True, help='do result visualization')
parser.add_argument('--debug_frame', type=int, default=-1, help='for debuging, input frame number')
parser.add_argument('--cfg', type=str, help='experiment configure file name')

args = parser.parse_args()
if args.cfg:
    update_config(args.cfg)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])

from core.base import Predictor

predictor = Predictor(args)
predictor(args.input, args.info, args.output, args.debug_frame)
