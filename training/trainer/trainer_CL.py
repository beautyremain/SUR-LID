# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: trainer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import pickle
import datetime
import logging
import numpy as np
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from metrics.base_metrics_class import Recorder
from torch.optim.swa_utils import AveragedModel, SWALR
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn import metrics

FFpp_pool = ['FaceForensics++', 'FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT']  #


class Trainer(object):
    def __init__(
            self,
            config,
            model,
            optimizer,
            scheduler,
            logger,
            metric_scoring='auc',
            time_now=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
            swa_model=None
    ):
        # check if all the necessary components are implemented
        if config is None or model is None or optimizer is None or logger is None:
            raise ValueError("config, model, optimizier, logger, and tensorboard writer must be implemented")

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.swa_model = swa_model
        self.writers = {}  # dict to maintain different tensorboard writers for each dataset and metric
        self.logger = logger
        self.metric_scoring = metric_scoring
        # maintain the best metric of all epochs
        self.best_metrics_all_time = defaultdict(
            lambda: defaultdict(lambda: float('-inf')
            if self.metric_scoring != 'eer' else float('inf'))
        )
        self.speed_up()  # move model to GPU

        # get current time
        self.timenow = config['time_now']
        # create directory path
        task_str = f"_{config['task_target']}" if config['task_target'] is not None else ""
        self.log_dir = os.path.join(
            self.config['log_dir'],
            self.config['model_name'] + task_str + '_' + self.timenow
        )
        os.makedirs(self.log_dir, exist_ok=True)

    def get_writer(self, phase, dataset_key, metric_key):
        writer_key = f"{phase}-{dataset_key}-{metric_key}"
        if writer_key not in self.writers:
            # update directory path
            writer_path = os.path.join(
                self.log_dir,
                phase,
                dataset_key,
                metric_key,
                "metric_board"
            )
            os.makedirs(writer_path, exist_ok=True)
            # update writers dictionary
            self.writers[writer_key] = SummaryWriter(writer_path)
        return self.writers[writer_key]

    def speed_up(self):

        self.model.to(device)
        self.model.device = device
        if self.config['ddp'] == True:
            num_gpus = torch.cuda.device_count()
            self.logger.info(f'avai gpus: {num_gpus}')
            # local_rank=[i for i in range(0,num_gpus)]
            self.model = DDP(self.model, device_ids=[self.config['local_rank']], find_unused_parameters=True,
                             output_device=self.config['local_rank'])
            # self.optimizer =  nn.DataParallel(self.optimizer, device_ids=[int(os.environ['LOCAL_RANK'])])

    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            suffix = model_path.split('.')[-1]
            if suffix == 'p':
                self.model.load_state_dict(saved.state_dict())
            else:
                self.model.load_state_dict(saved)
            self.logger.info('Model found in {}'.format(model_path))
        else:
            raise NotImplementedError(
                "=> no model found at '{}'".format(model_path))

    def save_ckpt(self, phase, dataset_key, ckpt_info=None,stage=None):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        if stage!=None:
            ckpt_name = f"ckpt_best_{stage}.pth"
        else:
            ckpt_name = f"ckpt_best.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        if self.config['ddp'] == True:
            torch.save(self.model.state_dict(), save_path)
        else:
            if 'svdd' in self.config['model_name']:
                torch.save({'R': self.model.R,
                            'c': self.model.c,
                            'state_dict': self.model.state_dict(), }, save_path)
            else:
                torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Checkpoint saved to {save_path}, current ckpt is {ckpt_info}")

    def save_swa_ckpt(self):
        save_dir = self.log_dir
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"swa.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        torch.save(self.swa_model.state_dict(), save_path)
        self.logger.info(f"SWA Checkpoint saved to {save_path}")

    def save_feat(self, phase, pred_dict, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        features = pred_dict['feat']
        feat_name = f"feat_best.npy"
        save_path = os.path.join(save_dir, feat_name)
        np.save(save_path, features.cpu().numpy())
        self.logger.info(f"Feature saved to {save_path}")

    def save_data_dict(self, phase, data_dict, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'data_dict_{phase}.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(data_dict, file)
        self.logger.info(f"data_dict saved to {file_path}")

    def save_metrics(self, phase, metric_one_dataset, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, 'metric_dict_best.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(metric_one_dataset, file)
        self.logger.info(f"Metrics saved to {file_path}")

    def train_step(self, data_dict):

        # pred1 = self.model.encoder_c(data_dict['image'])[0]
        # import pdb
        # pdb.set_trace()
        # self.model.loss_func['cls'](pred1, data_dict['label']).backward()

        if self.config['optimizer']['type'] == 'sam':
            for i in range(2):

                predictions = self.model(data_dict)
                losses = self.model.get_losses(data_dict, predictions)
                if i == 0:
                    pred_first = predictions
                    losses_firts = losses
                self.optimizer.zero_grad()
                losses['overall'].backward()
                if i == 0:
                    self.optimizer.first_step(zero_grad=True)
                else:
                    self.optimizer.second_step(zero_grad=True)
            return losses_firts, pred_first
        else:

            predictions = self.model(data_dict)
            if type(self.model) is DDP:
                losses = self.model.module.get_losses(data_dict, predictions)
            else:
                losses = self.model.get_losses(data_dict, predictions)
            self.optimizer.zero_grad()
            losses['overall'].backward()
            self.optimizer.step()

            return losses, predictions

    def train_epoch(
            self,
            epoch,
            train_data_loader,
            test_data_loaders=None,
            task_num=0
    ):

        self.logger.info("===> Epoch[{}] start!".format(epoch))
        test_best_metric=None
        if epoch >= 4:
            times_per_epoch = 3  # 是不是还是有点多了
        else:
            times_per_epoch = 1
        if self.config['ddp']:
            train_data_loader.sampler.set_epoch(epoch)
        test_step = len(train_data_loader) // times_per_epoch
        step_cnt = epoch * len(train_data_loader)

        # save the training data_dict
        data_dict = train_data_loader.dataset.data_dict
        self.save_data_dict('train', data_dict, ','.join(self.config['train_dataset']))
        # define training recorder
        train_recorder_loss = defaultdict(Recorder)
        train_recorder_metric = defaultdict(Recorder)

        for iteration, data_dict in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):
            self.setTrain()
            # more elegant and more scalable way of moving data to GPU
            for key in data_dict.keys():
                if data_dict[key] != None:
                    data_dict[key] = data_dict[key].cuda()

            losses, predictions = self.train_step(data_dict)
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            print(f" --------------------- 峰值显存使用: {peak_memory:.2f} MB -----------------",flush=True)
            if hasattr(self.model,'post_update'):
                updated_value=self.model.post_update(epoch=epoch)
                if updated_value is not None:
                    losses['post-updated']=updated_value
            # update learning rate
            # return None
            if self.config['SWA'] and epoch > self.config['swa_start']:
                self.swa_model.update_parameters(self.model)

            # compute training metric for each batch data
            if type(self.model) is DDP:
                batch_metrics = self.model.module.get_train_metrics(data_dict, predictions)
            else:
                batch_metrics = self.model.get_train_metrics(data_dict, predictions)

            # store data by recorder
            ## store metric
            for name, value in batch_metrics.items():
                train_recorder_metric[name].update(value)
            ## store loss
            for name, value in losses.items():
                train_recorder_loss[name].update(value)

            # run tensorboard to visualize the training process
            if iteration % 300 == 0 and self.config['local_rank'] == 0:
                if self.config['SWA'] and (epoch > self.config['swa_start'] or self.config['dry_run']):
                    self.scheduler.step()
                # info for loss
                loss_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_loss.items():
                    v_avg = v.average()
                    if v_avg == None:
                        loss_str += f"training-loss, {k}: not calculated"
                        continue
                    loss_str += f"training-loss, {k}: {v_avg}    "
                    # tensorboard-1. loss
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_loss_stage-{task_num}/{k}', v_avg, global_step=step_cnt)
                self.logger.info(loss_str)
                # info for metric
                metric_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_metric.items():
                    v_avg = v.average()
                    if v_avg == None:
                        metric_str += f"training-metric, {k}: not calculated    "
                        continue
                    metric_str += f"training-metric, {k}: {v_avg}    "
                    # tensorboard-2. metric
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_metric_stage-{task_num}/{k}', v_avg, global_step=step_cnt)
                self.logger.info(metric_str)

                # clear recorder.
                # Note we only consider the current 300 samples for computing batch-level loss/metric
                for name, recorder in train_recorder_loss.items():  # clear loss recorder
                    recorder.clear()
                for name, recorder in train_recorder_metric.items():  # clear metric recorder
                    recorder.clear()

            # run test
            if self.config['train_mode'] == 'debug' and "3060" in torch.cuda.get_device_name():
                count=1
            else:
                count=1

            if (step_cnt + count) % test_step == 0 and (epoch>=self.config.get('test_start',0) or self.config['train_mode']=='debug'):
                if test_data_loaders is not None and (not self.config['ddp']):
                    self.logger.info("===> Test start!")
                    test_best_metric = self.test_epoch(
                        epoch,
                        iteration,
                        test_data_loaders,
                        step_cnt,
                        current_train=train_data_loader.dataset.dataset_list,
                        task_num=task_num
                    )
                elif test_data_loaders is not None and (self.config['ddp'] and dist.get_rank() == 0):
                    test_best_metric = self.test_epoch(
                        epoch,
                        iteration,
                        test_data_loaders,
                        step_cnt,
                        current_train=train_data_loader.dataset.dataset_list,
                        task_num=task_num
                    )
                else:
                    test_best_metric = None
                # if self.config['ddp']:
                #     dist.barrier()

                # total_end_time = time.time()
            # total_elapsed_time = total_end_time - total_start_time
            # print("总花费的时间: {:.2f} 秒".format(total_elapsed_time))
            step_cnt += 1
            # if self.config['train_mode'] == 'debug':
            #     break
        return test_best_metric

    def get_respect_acc(self, prob, label):
        pred = np.where(prob > 0.5, 1, 0)
        judge = (pred == label)
        zero_num = len(label) - np.count_nonzero(label)
        acc_fake = np.count_nonzero(judge[zero_num:]) / len(judge[zero_num:])
        acc_real = np.count_nonzero(judge[:zero_num]) / len(judge[:zero_num])
        return acc_real, acc_fake

    def test_one_dataset(self, data_loader):
        # define test recorder
        test_recorder_loss = defaultdict(Recorder)
        prediction_lists = []
        label_lists = []
        for i, data_dict in enumerate(data_loader):
            # get data
            if 'label_spe' in data_dict:
                data_dict.pop('label_spe')  # remove the specific label
            data_dict['label'] = torch.where(data_dict['label'] != 0, 1, 0)  # fix the label to 0 and 1 only
            # move data to GPU
            for key in data_dict.keys():
                if data_dict[key] != None:
                    data_dict[key] = data_dict[key].cuda()
            # model forward without considering gradient computation
            predictions = self.inference(data_dict)
            label_lists += list(data_dict['label'].cpu().detach().numpy())
            prediction_lists += list(predictions['prob'].cpu().detach().numpy())
            if type(self.model) is not AveragedModel:
                # compute all losses for each batch data
                if type(self.model) is DDP:
                    losses = self.model.module.get_losses(data_dict, predictions)
                else:
                    losses = self.model.get_losses(data_dict, predictions)

                # store data by recorder
                for name, value in losses.items():
                    test_recorder_loss[name].update(value)

        return test_recorder_loss, np.array(prediction_lists), np.array(label_lists)

    def get_test_metrics(self, y_pred, y_true, img_names=None):

        def get_video_metrics(image, pred, label):
            result_dict = {}
            new_label = []
            new_pred = []

            for item in np.transpose(np.stack((image, pred, label)), (1, 0)):
                # 分割字符串，获取'a'和'b'的值
                s = item[0]
                try:
                    parts = s.split('\\')
                    a = parts[-2]
                    b = parts[-1]
                except Exception as e:
                    parts = s.split('/')
                    a = parts[-2]
                    b = parts[-1]

                # 如果'a'的值还没有在字典中，添加一个新的键值对
                if a not in result_dict:
                    result_dict[a] = []

                # 将'b'的值添加到'a'的列表中
                result_dict[a].append(item)
            image_arr = list(result_dict.values())
            # 将字典的值转换为一个列表，得到二维数组

            for video in image_arr:
                pred_sum = 0
                label_sum = 0
                len = 0
                for frame in video:
                    pred_sum += float(frame[1])
                    label_sum += int(frame[2])
                    len += 1
                new_pred.append(pred_sum / len)
                new_label.append(int(label_sum / len))
            fpr, tpr, thresholds = metrics.roc_curve(new_label, new_pred)
            v_auc = metrics.auc(fpr, tpr)
            fnr = 1 - tpr
            v_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
            return v_auc, v_eer

        # auc
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # eer
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        # ap
        ap = metrics.average_precision_score(y_true, y_pred)
        # acc
        prediction_class = (y_pred > 0.5).astype(int)
        correct = (prediction_class == np.clip(y_true, a_min=0, a_max=1)).sum().item()
        acc = correct / len(prediction_class)
        # video auc
        try:
            v_auc, _ = get_video_metrics(img_names, y_pred, y_true)
        except Exception as e:
            v_auc=1

        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'pred': y_pred, 'v_auc': v_auc, 'label': y_true}

    def save_best(self, epoch, iteration, step, losses_one_dataset_recorder, key, metric_one_dataset,task_num=0):
        best_metric = self.best_metrics_all_time[key].get(self.metric_scoring,
                                                          float('-inf') if self.metric_scoring != 'eer' else float(
                                                              'inf'))
        # Check if the current score is an improvement
        improved = (metric_one_dataset[self.metric_scoring] > best_metric) if self.metric_scoring != 'eer' else (
                metric_one_dataset[self.metric_scoring] < best_metric)
        if improved and epoch >= self.config['save_start']:
            # Update the best metric
            self.best_metrics_all_time[key][self.metric_scoring] = metric_one_dataset[self.metric_scoring]
            if key == 'avg':
                self.best_metrics_all_time[key]['dataset_dict'] = metric_one_dataset['dataset_dict']
            # Save checkpoint, feature, and metrics if specified in config
            if (self.config['save_ckpt'] and key not in FFpp_pool and  key!='avg'):
                self.save_ckpt('test', key, f"{epoch}+{iteration}")
            if key == 'avg':
                self.save_ckpt('test', key, f"{epoch}+{iteration}",stage=task_num)
            self.save_metrics('test', metric_one_dataset, key)
        if losses_one_dataset_recorder is not None:
            # info for each dataset
            loss_str = f"dataset: {key}    step: {step}    "
            for k, v in losses_one_dataset_recorder.items():
                writer = self.get_writer('test', key, k)
                v_avg = v.average()
                if v_avg == None:
                    print(f'{k} is not calculated')
                    continue
                # tensorboard-1. loss
                writer.add_scalar(f'test_losses-{task_num}/{k}', v_avg, global_step=step)
                loss_str += f"testing-loss, {k}: {v_avg}    "
            self.logger.info(loss_str)
        # tqdm.write(loss_str)
        metric_str = f"dataset: {key}    step: {step}    "
        for k, v in metric_one_dataset.items():
            if k == 'pred' or k == 'label' or k == 'dataset_dict':
                continue
            metric_str += f"testing-metric, {k}: {v}    "
            # tensorboard-2. metric
            writer = self.get_writer('test', key, k)
            writer.add_scalar(f'test_metrics-{task_num}/{k}', v, global_step=step)
        if 'pred' in metric_one_dataset:
            acc_real, acc_fake = self.get_respect_acc(metric_one_dataset['pred'], metric_one_dataset['label'])
            metric_str += f'testing-metric, acc_real:{acc_real}; acc_fake:{acc_fake}'
            writer.add_scalar(f'test_metrics-{task_num}/acc_real', acc_real, global_step=step)
            writer.add_scalar(f'test_metrics-{task_num}/acc_fake', acc_fake, global_step=step)
        self.logger.info(metric_str)

    def test_epoch(self, epoch, iteration, test_data_loaders, step,current_train=None,task_num=0):
        # set model to eval mode
        self.setEval()

        # define test recorder
        losses_all_datasets = {}
        metrics_all_datasets = {}
        best_metrics_per_dataset = defaultdict(dict)  # best metric for each dataset, for each metric
        avg_metric = {'acc': 0, 'auc': 0, 'eer': 0, 'ap': 0, 'video_auc': 0, 'dataset_dict': {}}
        # testing for all test data
        keys = test_data_loaders.keys()
        if current_train is not None and len(current_train) == 1:
            avg_stop=list(keys).index(current_train[0])
        else:
            avg_stop=len(keys)-1
        for j, key in enumerate(keys):
            # save the testing data_dict
            data_dict = test_data_loaders[key].dataset.data_dict
            self.save_data_dict('test', data_dict, key)

            # compute loss for each dataset
            losses_one_dataset_recorder, predictions_nps, label_nps = self.test_one_dataset(test_data_loaders[key])
            # print(f'stack len:{predictions_nps.shape};{label_nps.shape};{len(data_dict["image"])}')
            losses_all_datasets[key] = losses_one_dataset_recorder
            metric_one_dataset = self.get_test_metrics(y_pred=predictions_nps, y_true=label_nps,
                                                       img_names=data_dict['image'])
            if j<=avg_stop:
                for metric_name, value in metric_one_dataset.items():
                    if metric_name in avg_metric:
                        avg_metric[metric_name] += value
                avg_metric['dataset_dict'][key] = metric_one_dataset[self.metric_scoring]
            if type(self.model) is AveragedModel:
                metric_str = f"Iter Final for SWA:    "
                for k, v in metric_one_dataset.items():
                    metric_str += f"testing-metric, {k}: {v}    "
                self.logger.info(metric_str)
                continue
            self.save_best(epoch, iteration, step, losses_one_dataset_recorder, key, metric_one_dataset,task_num=task_num)

        if len(keys) > 0 and self.config.get('save_avg', False):
            # calculate avg value
            for key in avg_metric:
                if key != 'dataset_dict':
                    avg_metric[key] /= (avg_stop+1)
            self.save_best(epoch, iteration, step, None, 'avg', avg_metric,task_num=task_num)

        self.logger.info('===> Test Done!')
        return self.best_metrics_all_time  # return all types of mean metrics for determining the best ckpt

    @torch.no_grad()
    def inference(self, data_dict):
        predictions = self.model(data_dict, inference=True)
        return predictions
