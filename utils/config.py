from easydict import EasyDict
from pathlib import Path
import yaml
import numpy as np
import argparse
import os

def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            logger.info('\n%s.%s = edict()' % (pre, key))
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.info('%s.%s: %s' % (pre, key, val))


def cfg_from_list(cfg_list, config):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = config
        for subkey in key_list[:-1]:
            assert subkey in d, 'NotFoundKey: %s' % subkey
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'NotFoundKey: %s' % subkey
        try:
            value = literal_eval(v)
        except:
            value = v

        if type(value) != type(d[subkey]) and isinstance(d[subkey], EasyDict):
            key_val_list = value.split(',')
            for src in key_val_list:
                cur_key, cur_val = src.split(':')
                val_type = type(d[subkey][cur_key])
                cur_val = val_type(cur_val)
                d[subkey][cur_key] = cur_val
        elif type(value) != type(d[subkey]) and isinstance(d[subkey], list):
            val_list = value.split(',')
            for k, x in enumerate(val_list):
                val_list[k] = type(d[subkey][0])(x)
            d[subkey] = val_list
        else:
            assert type(value) == type(d[subkey]), \
                'type {} does not match original type {}'.format(type(value), type(d[subkey]))
            d[subkey] = value


def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key in ['MAX_VOLUME_SPACE', 'MIN_VOLUME_SPACE']:
                if val[1] == 'PI':
                    val[1] = np.pi
                elif val[1] == '-PI':
                    val[1] = -np.pi
            if isinstance(key, int):
                key = str(key)
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config


def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
        merge_new_config(config=config, new_config=new_config)
    return config

cfg = EasyDict()
cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()

def parse_config():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default='cfgs/default.yaml')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_sche', action='store_true', default=False)
    parser.add_argument('--npoints', type=int, default=4096)
    parser.add_argument('--output_dir', type=str, default='./output/smpl_n1')
    parser.add_argument('--ckpt_name', type=str, default='model.ckpt')
    parser.add_argument('--launcher', type=str, default='slurm')
    parser.add_argument('--tcp_port', type=int, default=12345)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist_train', action='store_true', default=False)
    parser.add_argument('--syncbn', type=int, default=0)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--vis', action='store_true', default=False)
    parser.add_argument('--T', type=int, default=20)

    parser.add_argument('--GarmentPCA', type=int, default=0)
    parser.add_argument('--GarmentPCALBS', type=int, default=0)
    parser.add_argument('--GarmentPCA_pretrain', type=str, default=None)
    parser.add_argument('--fix_PCA', type=int, default=0)
    parser.add_argument('--only_seg', type=int, default=0)
    parser.add_argument('--MGN', type=int, default=0)

    args = parser.parse_args()
    cfg_from_yaml_file(args.config, cfg)

    cfg.GARMENT.TEMPLATE = os.path.join(cfg.DATASET.ROOT_FOLDER, cfg.DATASET.GARMENT_FOLDER, cfg.GARMENT.TEMPLATE)
    cfg.GARMENT.PCACOMPONENTSFILE = os.path.join(cfg.DATASET.ROOT_FOLDER, cfg.DATASET.GARMENT_FOLDER, cfg.GARMENT.PCACOMPONENTSFILE)
    cfg.DATASET.SMPL_PARAM_PREFIX = os.path.join(cfg.DATASET.ROOT_FOLDER, cfg.DATASET.CLOTH3D_FOLDER)
    cfg.DATASET.GARMENT_TEMPLATE_T_POSE_PREFIX = os.path.join(cfg.DATASET.ROOT_FOLDER, cfg.DATASET.GARMENT_TEMPLATE_T_POSE_PREFIX)

    return args, cfg

args, cfg = parse_config()
