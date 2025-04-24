# author: Jikang Cheng
# date: 2024-11
# description: training code.


# Synthetic
import getpass
import os
from collections import defaultdict

from torch.optim.swa_utils import AveragedModel, SWALR
from optimizor.LinearLR import LinearDecayLR
from optimizor.SAM import SAM
from utils.tools import print_cpu_gpu_usage

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset

from trainer.trainer_CL import Trainer
from detectors import DETECTOR

import argparse
from logger import create_logger, RankFilter
import warnings
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import pdb
import pdb

# pdb.set_trace()


warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str,
                    default='/home/zhiyuanyan/DeepfakeBench/training/config/detector/meso4Inception.yaml',
                    help='path to detector YAML file')
parser.add_argument("--train_dataset", nargs="+")
parser.add_argument('--data_manner', type=str,
                    default='img',
                    help='from raw images (img) or LMDB (lmdb)')
parser.add_argument('--task_folder', type=str,
                    default=None)
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument("--ddp", action='store_true', default=False)
parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=True)
parser.add_argument('--no-save_feat', dest='save_feat', action='store_false', default=True)
parser.add_argument('--batch_task', action='store_true', default=False)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--task_target', type=str,
                    default=None,
                    help='the purpose of training, and saving accordingly.')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--seed', type=int, default=-1)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
on_2060 = "2060" in torch.cuda.get_device_name()
on_3060 = "3060" in torch.cuda.get_device_name()
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def prepare_training_data(config):
    def load_training_dataset(config, name=None):
        if name is not None:
            config = config.copy()
            config['train_dataset'] = name
        # Only use the blending dataset class in training
        if 'dataset_type' in config and config['dataset_type'] == 'blend':
            if config['model_name'] == 'facexray':
                train_set = FFBlendDataset(config)
            elif config['model_name'] == 'fwa' or config['model_name'] == 'dsp_fwa':
                train_set = FWABlendDataset(config)
            else:
                raise NotImplementedError(
                    'Only facexray, fwa, and dsp_fwa are currently supported for blending dataset'
                )
        elif 'svdd_' in config['model_name']:
            train_set_all = DeepfakeAbstractBaseDataset(
                config=config,
                mode='train',
            )
            train_set = train_set_all.sub_set
            return torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                shuffle=True,
                num_workers=int(config['workers']),
                collate_fn=train_set_all.collate_fn,
                drop_last=False
            )
        else:
            train_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='train',
            )
        if config['ddp']:
            sampler = DistributedSampler(train_set)
            train_data_loader = \
                torch.utils.data.DataLoader(
                    dataset=train_set,
                    batch_size=config['train_batchSize'] // torch.cuda.device_count(),
                    num_workers=int(config['workers']),
                    collate_fn=train_set.collate_fn,
                    sampler=sampler
                )
        else:
            train_data_loader = \
                torch.utils.data.DataLoader(
                    dataset=train_set,
                    batch_size=config['train_batchSize'],
                    shuffle=True,
                    num_workers=int(config['workers']),
                    collate_fn=train_set.collate_fn,
                    drop_last=False
                )

        return train_data_loader

    if config['data_im'] == 'joint':
        return load_training_dataset(config)
    else:
        train_dataset_loaders = {}
        for one_train_name in config['train_dataset']:
            train_dataset_loaders[one_train_name] = load_training_dataset(config, [one_train_name])
        return train_dataset_loaders


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        if not config.get('dataset_type', None) == 'lrl':
            test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test',
            )
        else:
            test_set = LRLDataset(
                config=config,
                mode='test',
            )

        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=config['test_batchSize'],
                shuffle=False,
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False  # (test_name=='DeepFakeDetection'),
            )

        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_optimizer(model, config):
    opt_name = config['optimizer']['type']
    if opt_name == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
            weight_decay=config['optimizer'][opt_name]['weight_decay']
        )
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )

    elif opt_name == 'sam':
        optimizer = SAM(model.parameters(), torch.optim.SGD,
                        lr=config['optimizer'][opt_name]['lr'], momentum=config['optimizer'][opt_name]['momentum'])
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(config['optimizer']))
    return optimizer


def choose_scheduler(config, optimizer):
    if config['SWA']:
        return SWALR(optimizer, swa_lr=config['swa_lr'])
    if config['lr_scheduler'] is None or config['lr_scheduler'] == 'None':
        return None
    elif config['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['lr_step'],
            gamma=config['lr_gamma'],
        )
    elif config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['lr_T_max'],
            eta_min=config['lr_eta_min'],
        )

    elif config['lr_scheduler'] == 'linear':
        scheduler = LinearDecayLR(optimizer, config['nEpochs'], int(config['nEpochs'] / 2))
    else:
        raise NotImplementedError('Scheduler {} is not implemented'.format(config['lr_scheduler']))
    return scheduler


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def init_dist(backend: str = 'nccl', **kwargs) -> None:
    num_gpus = torch.cuda.device_count()
    print('num gpus: ', num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def parse_metric_for_print(metric_dict):
    if metric_dict is None:
        return "\n"
    str = "\n"
    str += "================================ Each dataset best metric ================================ \n"
    for key, value in metric_dict.items():
        if key != 'avg':
            str = str + f"| {key}: "
            for k, v in value.items():
                str = str + f" {k}={v} "
            str = str + "| \n"
        else:
            str += "============================================================================================= \n"
            str += "================================== Average best metric ====================================== \n"
            avg_dict = value
            for avg_key, avg_value in avg_dict.items():
                if avg_key == 'dataset_dict':
                    for key, value in avg_value.items():
                        str = str + f"| {key}: {value} | \n"
                else:
                    str = str + f"| avg {avg_key}: {avg_value} | \n"
    str += "============================================================================================="
    return str

def continual_step(config,model,train_dataloader,counter,logger=None):
    torch.cuda.empty_cache()
    logger.info('resetting optimizer')
    optimizer=choose_optimizer(model,config)
    scheduler=choose_scheduler(config,optimizer)
    logger.info(optimizer)
    if logger is not None:
        logger.info(f'Current Counter: {counter}')
    if hasattr(model, 'update_CL'):
        if logger is not None:
            logger.info('update CL start')
        print_cpu_gpu_usage()
        model.update_CL([train_dataloader,model,config],counter)
        print_cpu_gpu_usage()
        if logger is not None:
            logger.info('update CL complete')

    else:
        print("The model does not contain the method 'update_CL'.")
    return optimizer,scheduler


def main():
    print('start_training',flush=True)
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/train_config_CL.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)

    config['backbone_config']['Gating'] = config['Gating']
    # If arguments are provided, they will overwrite the yaml settings
    if args.train_dataset:
        config['train_dataset'] = args.train_dataset
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    config['detect_method'] = os.path.split(args.detector_path)[-1][:-5]
    config['task_target'] = args.task_target
    print(config['task_target'])
    if args.seed > -1:
        config['manualSeed'] = args.seed
        config['task_target'] = config['task_target'] + f"_seed{config['manualSeed']}"
    config['data_manner'] = args.data_manner
    if args.data_manner == 'lmdb':
        config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'
    if config['dry_run']:
        config['nEpochs'] = 0
        config['save_feat'] = False
        config['swa_start'] = 0
    config['local_rank'] = args.local_rank
    # config['save_ckpt'] = args.save_ckpt
    # config['save_feat'] = args.save_feat
    config['ddp'] = args.ddp
    config['task_folder'] = args.task_folder
    time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    config['time_now']=time_now
    # create logger
    config[
        'log_dir'] = f"logs/training/{config['model_name'] if config['model_name'] != 'efficientnetb4' else 'effnb4'}"
    if args.task_folder != None:
        config['log_dir'] = config['log_dir'] + "/" + args.task_folder
    task_str = f"_{config['task_target']}" if config['task_target'] is not None else ""
    logger_path = os.path.join(
        config['log_dir'],
        config['model_name'] + task_str + '_' + config['time_now']
    )
    os.makedirs(logger_path, exist_ok=True)
    if on_2060:
        config['lmdb_dir'] = r'I:\transform_2_lmdb'
        config['train_batchSize'] = 1
        config['workers'] = 0
    elif on_3060:
        config['lmdb_dir'] = r'F:\Datasets\lmdb'
        config['train_batchSize'] = 4
        config['workers'] = 0
    else:
        config['workers'] = 8

        username = getpass.getuser()
        print(f'username: {username}', flush=True)
        if username == 'zywang4':
            config['lmdb_dir'] = r'/scratch/zywang4/cjk/data/lmdb/'
        elif username == 'chengjikang':
            config['lmdb_dir'] = r'/project/chengjikang/data/lmdb/'

    logger = create_logger(os.path.join(logger_path, "training.log"))
    logger.info('Save log to {}'.format(logger_path))
    if args.debug:
        config['nEpochs']=0
        config['train_mode'] = 'debug'
    # print configuration
    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    logger.info(params_string)

    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if config['ddp']:
        dist.init_process_group(backend='gloo')
        logger.addFilter(RankFilter(0))
    # print('local rank:  ', args.local_rank, os.environ['LOCAL_RANK'])

    # prepare the training data loader
    train_data_loaders = prepare_training_data(config)

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)

    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config)
    if 'restore_ckpt' in config:

        if config['restore_ckpt'] != "None":
            ckpt = torch.load(config['restore_ckpt'], map_location=device)
            model.load_state_dict(ckpt, strict=True)
    # prepare the optimizer
    optimizer = choose_optimizer(model, config)

    # prepare the scheduler
    scheduler = choose_scheduler(config, optimizer)

    # prepare the metric
    metric_scoring = choose_metric(config)

    if config['SWA']:
        swa_model = AveragedModel(model).cuda()
    else:
        swa_model = None

    # prepare the trainer
    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring, swa_model=swa_model,
                      time_now=time_now)

    test_set_num=len(train_data_loaders)

    def create_defaultdict():
        return defaultdict(lambda: defaultdict(lambda: float('-inf') if metric_scoring != 'eer' else float('inf')))

    # 使用列表生成器来创建4个独立的defaultdict对象
    # dict_list = [create_defaultdict() for _ in range(4)]
    one_step_metric_collectors=[create_defaultdict().copy() for _ in range(4)]
    # if 'svdd' in config['model_name']:
    #     if '2c' in config['model_name']:
    #         model.init_c_r(train_data_loader.dataset.subsets()) # ,torch.load('cs_dict.pt')
    #     else:
    #         model.init_c_r(train_data_loader)
    counter=1
    task_num=0
    for train_data_loader,one_step_metric_collector in zip(train_data_loaders.values(),one_step_metric_collectors):
        trainer.best_metrics_all_time = one_step_metric_collector
        # start training
        for epoch in range(config['start_epoch'], config['nEpochs'] + 1):
            trainer.model.epoch = epoch
            # if counter==0 and on_3060:
            #     continual_step(config, optimizer, model,train_data_loader,counter)
            best_metric = trainer.train_epoch(
                epoch=epoch,
                train_data_loader=train_data_loader,
                test_data_loaders=test_data_loaders,
                task_num=task_num
            )
            if best_metric is not None:
                logger.info(
                    f"===> Epoch[{epoch}] end with testing {metric_scoring}: {parse_metric_for_print(best_metric)}")
            else:
                logger.info(
                    f"===> Epoch[{epoch}] end without testing")
            if 'svdd' in config['model_name']:
                model.update_R(epoch)
            if scheduler is not None:
                scheduler.step()
            if config['train_mode']=='debug':
                break
        if config['SWA']:
            pass
        if best_metric is not None:
            logger.info(f"===> Current finished training dataset are: {train_data_loader.dataset.dataset_list}.")
            logger.info(f"It ends with testing {metric_scoring}: {parse_metric_for_print(best_metric)}.")
        optimizer,scheduler=continual_step(config,model,train_data_loader,counter,logger)
        counter = counter + 1
        task_num+=1
    logger.info(f"===> Entire continual training is finished.")
    if hasattr(model, 'save_replay'):
        replay_path = os.path.join(logger_path,'replays')
        if not os.path.exists(replay_path):
            os.makedirs(replay_path)
        if logger is not None:
            logger.info(f'saving replay to {replay_path}')
            model.save_replay(replay_path)
    for train_data_loader, one_step_metric_collector in zip(train_data_loaders.values(), one_step_metric_collectors):
        logger.info(f"===> Training dataset: {train_data_loader.dataset.dataset_list}.")
        logger.info(f"It ends with testing {metric_scoring}: {parse_metric_for_print(one_step_metric_collector)}.")


    # close the tensorboard writers
    for writer in trainer.writers.values():
        writer.close()


if __name__ == '__main__':
    main()
