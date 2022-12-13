import os
import sys
import time
import math
import numpy as np
import cv2
import shutil
import os.path as osp
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.nn import functional as F
import matplotlib.pyplot as plt

from core.config import cfg

def get_images(file_name, tmp_path, debug=False):
        cap = cv2.VideoCapture(file_name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        os.makedirs(tmp_path, exist_ok=True)

        width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if width > 800:
            height = int(height * 800 / width)
            width = 800
        elif height > 450:
            width = int(width * 450 / height)
            height = 450

        idx = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if ret == False:
                break

            frame = cv2.resize(frame, (width, height))
            cv2.imwrite(osp.join(tmp_path, '{0:09d}.jpg'.format(idx)),frame)
            idx += 1


            if debug and (idx == 500):
                print("\n==> Debug mode")
                break

        cap.release()
        cv2.destroyAllWindows()
        del cap
        return idx, fps

def select_target_id(results):
    areas = []

    for result in results:
        bbox = result['bbox']
        area = (bbox[:,2] * bbox[:,3]).mean()
        areas.append(area)
    
    areas = np.array(areas)
    return np.argmax(areas)

def annToMask(segm, h, w):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = mask_util.frPyObjects(segm, h, w)
        rle = mask_util.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = mask_util.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann['segmentation']

    mask = mask_util.decode(rle)
    return mask


def sample_image_feature(img_feat, xy, width, height):
    x = xy[:,0] / width * 2 - 1
    y = xy[:,1] / height * 2 - 1
    grid = torch.stack((x,y),1)[None,:,None,:]
    img_feat = F.grid_sample(img_feat, grid, align_corners=True)[0,:,:,0] # (channel_dim, sampling points)
    img_feat = img_feat.permute(1,0)
    return img_feat


def lr_check(optimizer, epoch):
    base_epoch = 5
    if False and epoch <= base_epoch:
        lr_warmup(optimizer, cfg.TRAIN.lr, epoch, base_epoch)

    for param_group in optimizer.param_groups:
        curr_lr = param_group['lr']
    print(f"Current epoch {epoch}, lr: {curr_lr}")


def lr_warmup(optimizer, lr, epoch, base):
    lr = lr * (epoch / base)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        self.acc += time.time() - self.t0  # cacluate time diff

    def reset(self):
        self.acc = 0

    def print(self):
        return round(self.acc, 2)


def stop():
    sys.exit()


def check_data_pararell(train_weight):
    new_state_dict = OrderedDict()
    for k, v in train_weight.items():
        name = k[7:]  if k.startswith('module') else k  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_optimizer(model):
    total_params = []
    for module in model.trainable_modules:
        total_params += list(module.parameters())

    optimizer = None
    if cfg.TRAIN.optimizer == 'sgd':
        optimizer = optim.SGD(
            total_params,
            lr=cfg.TRAIN.lr,
            momentum=cfg.TRAIN.momentum,
            weight_decay=cfg.TRAIN.weight_decay,
            nesterov=cfg.TRAIN.nesterov
        )
    elif cfg.TRAIN.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(
            total_params,
            lr=cfg.TRAIN.lr
        )
    elif cfg.TRAIN.optimizer == 'adam':
        optimizer = optim.Adam(
            total_params,
            lr=cfg.TRAIN.lr
        )
    elif cfg.TRAIN.optimizer == 'adamw':
        optimizer = optim.AdamW(
            total_params,
            lr=cfg.TRAIN.lr,
            weight_decay=0.1
        )

    return optimizer


def get_scheduler(optimizer):
    scheduler = None
    if cfg.TRAIN.scheduler == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.lr_step, gamma=cfg.TRAIN.lr_factor)
    elif cfg.TRAIN.scheduler == 'platue':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.TRAIN.lr_factor, patience=10, min_lr=1e-5)

    return scheduler


def save_checkpoint(states, epoch, is_best=None):
    file_name = f'epoch_{epoch}.pth.tar'
    output_dir = cfg.checkpoint_dir
    if states['epoch'] == cfg.TRAIN.end_epoch:
        file_name = 'final.pth.tar'
    torch.save(states, os.path.join(output_dir, file_name))

    if is_best:
        torch.save(states, os.path.join(output_dir, 'best.pth.tar'))


def load_checkpoint(load_dir, epoch=0, pick_best=False):
    try:
        print(f"Fetch model weight from {load_dir}")
        checkpoint = torch.load(load_dir, map_location='cuda')
        return checkpoint
    except Exception as e:
        raise ValueError("No checkpoint exists!\n", e)


def save_plot(data_list, epoch, title='Train Loss'):
    f = plt.figure()

    plot_title = '{} epoch {}'.format(title, epoch)
    file_ext = '.pdf'
    save_path = '_'.join(title.split(' ')).lower() + file_ext

    plt.plot(np.arange(1, len(data_list) + 1), data_list, 'b-', label=plot_title)
    plt.legend()
    plt.title(plot_title)
    plt.xlabel('epoch')
    plt.xlim(left=0, right=len(data_list) + 1)
    plt.xticks(np.arange(0, len(data_list) + 1, 1.0), fontsize=5)

    min_value = np.asarray(data_list).min()
    plt.annotate('%0.2f' % min_value, xy=(1, min_value), xytext=(8, 0),
                 arrowprops=dict(arrowstyle="simple", connectionstyle="angle3"),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')

    f.savefig(os.path.join(cfg.graph_dir, save_path))
    plt.close(f)
