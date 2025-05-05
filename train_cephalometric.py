#!/usr/bin/env python
import os
import argparse
import torch
import mmcv
from mmcv.config import Config
from mmcv import DictAction
from mmcv.runner import init_dist, set_random_seed
from mmpose.apis import train_model
from mmpose.datasets import build_dataset
from mmpose.models import build_posenet
from mmpose.utils import collect_env, get_root_logger, setup_multi_processes

def parse_args():
    parser = argparse.ArgumentParser(description='Train a pose model')
    parser.add_argument('--config', 
                      default='cephalometric_hrnetv2_w18_config.py',
                      help='train config file path')
    parser.add_argument('--work-dir', 
                      default='work_dirs/cephalometric',
                      help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    
    parser.add_argument(
        '--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = os.path.join('./work_dirs',
                                    os.path.splitext(os.path.basename(args.config))[0])
    
    # create work_dir
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    
    # resume training
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    
    if args.no_validate:
        cfg.evaluation.interval = 0

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    seed = args.seed
    if seed is not None:
        print(f'Set random seed to {seed}, deterministic: '
              f'{args.deterministic}')
        set_random_seed(seed, deterministic=args.deterministic)

    # build the dataset
    datasets = [build_dataset(cfg.data.train)]

    # build the model and load pretrained weights
    model = build_posenet(cfg.model)
    
    # create a logger to log training information
    timestamp = mmcv.utils.get_time_str()
    log_file = os.path.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # start training
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=None)


if __name__ == '__main__':
    main() 