import os
import os.path as osp
import shutil

import yaml
from easydict import EasyDict as edict
import datetime


def init_dirs(dir_list):
    for dir in dir_list:
        if os.path.exists(dir) and os.path.isdir(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)


cfg = edict()

""" Directory """
cfg.cur_dir = osp.dirname(os.path.abspath(__file__))
cfg.root_dir = osp.join(cfg.cur_dir, '../../')
cfg.data_dir = osp.join(cfg.root_dir, 'data')
cfg.smpl_dir = osp.join(cfg.root_dir, 'smplpytorch')
KST = datetime.timezone(datetime.timedelta(hours=9))
save_folder = 'exp_' + str(datetime.datetime.now(tz=KST))[5:-16]
save_folder = save_folder.replace(" ", "_")
save_folder_path = 'experiment/{}'.format(save_folder)

""" Dataset """
cfg.DATASET = edict()
cfg.DATASET.workers = 16
cfg.DATASET.batch_size = 8
cfg.DATASET.min_frame_ratio = 0.33
cfg.DATASET.bbox_scale = 1.2
cfg.DATASET.default_information = osp.join(cfg.cur_dir, 'default_information.json')


""" Model """
cfg.MODEL = edict()
cfg.MODEL.input_shape = (224, 224)


""" SPIN """
cfg.SPIN = edict()
cfg.SPIN.spin_dir = osp.join(cfg.root_dir, 'lib', 'SPIN')
cfg.SPIN.SMPL_MEAN_PARAMS = osp.join(cfg.SPIN.spin_dir, 'data', 'smpl_mean_params.npz')
cfg.SPIN.checkpoint = osp.join(cfg.SPIN.spin_dir, 'data', 'model_checkpoint.pt')
cfg.SPIN.SMPL_MODEL_DIR = osp.join(cfg.SPIN.spin_dir, 'data', 'smpl')
cfg.SPIN.FOCAL_LENGTH = 5000
cfg.SPIN.IMG_RES = 224


""" Augmentation """
cfg.AUG = edict()
cfg.AUG.flip = False
cfg.AUG.rotate_factor = 0  # 30


""" Test Detail """
cfg.TEST = edict()


def _update_dict(k, v):
    for vk, vv in v.items():
        if vk in cfg[k]:
            cfg[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in cfg:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        cfg[k][0] = (tuple(v))
                    else:
                        cfg[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


