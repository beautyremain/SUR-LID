import copy
import os
import datetime
import logging
import numpy as np
import yaml
from fvcore.nn import FlopCountAnalysis
from sklearn import metrics
from typing import Union
import math
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors.base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
import random
from detectors.utils.dfil_api import loss_FD,loss_fn_kd
from detectors.utils.lid_api import *

logger = logging.getLogger(__name__)
semi_frozen_alpha=[0.99,0.999,0.9999,0.99995]
frozen_alpha=[0.99,1,1,1]
@DETECTOR.register_module(module_name='CL_Lid')
class CL_LIDEfficientDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.teacher = self.build_backbone(config)
        self.teacher_classifier=None
        self.loss_func = self.build_loss(config)
        self.enable_teacher=False
        self.continual_index=0
        # self.memory_dict={'Dataset1':{'Real':{'Center_feat':None,'Images':[]},'Fake':None}}
        self.real_memory_array=[]
        self.fake_memory_array=[]
        self.real_center_array=[]
        self.fake_center_array=[]
        self.save_script()
        self.fc_groups=[LogisticRegression(1792, 2).cuda() for _ in range(4)]

    def build_backbone(self, config):

        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        model_config['pretrained'] = self.config['pretrained']
        backbone = backbone_class(model_config)
        if config['pretrained'] != 'None':
            logger.info('Load pretrained model successfully!')
        else:
            logger.info('No pretrained model.')

        return backbone
    def post_update(self,epoch=0):
        if epoch>self.config['align_epoch']:
            return self.update_weight()

    def update_weight(self):
        if self.config.get('frozen','semi'):
            alphas = semi_frozen_alpha
        else:
            alphas = frozen_alpha
        if self.continual_index==0:
            return None
        elif self.continual_index==1:
            return self.update_weight_two(alphas)
        elif self.continual_index==2:
            return self.update_weight_three(alphas)
        elif self.continual_index==3:
            return self.update_weight_four(alphas)

    def update_weight_two(self,alpha=[0.99,0.999]):
        beta0 = normalizer(self.fc_groups[0].weight.data.squeeze())
        beta1 = normalizer(self.fc_groups[1].weight.data.squeeze())
        # 归一化计算
        self.fc_groups[0].weight.data = (normalizer(alpha[1]  * beta0 + (1 - alpha[1]) * beta1) * torch.norm(self.fc_groups[0].weight.data, dim=-1, keepdim=True))#.unsqueeze(0) 这玩意在为1时才要
        self.fc_groups[1].weight.data = (normalizer(alpha[0] * beta1 + (1 - alpha[0]) * beta0) * torch.norm(self.fc_groups[1].weight.data, dim=-1, keepdim=True))

        return (beta0 * beta1).sum()

    def update_weight_three(self,alpha=[0.99,0.999,0.9999]):
        beta0 = normalizer(self.fc_groups[0].weight.data.squeeze())
        beta1 = normalizer(self.fc_groups[1].weight.data.squeeze())
        beta2 = normalizer(self.fc_groups[2].weight.data.squeeze())

        beta0_fr =  min([((beta0 * beta1).sum().item(), 0, beta1), ((beta0 * beta2).sum().item(), 1, beta2)])[2]
        beta1_fr =  min([((beta1 * beta0).sum().item(), 0, beta0), ((beta1 * beta2).sum().item(), 1, beta2)])[2]
        beta2_fr =  min([((beta2 * beta0).sum().item(), 0, beta0), ((beta2 * beta1).sum().item(), 1, beta1)])[2]

        self.fc_groups[0].weight.data = (normalizer(alpha[2] * beta0 + (1 - alpha[2]) * beta0_fr) * torch.norm(self.fc_groups[0].weight.data, dim=-1, keepdim=True))
        self.fc_groups[1].weight.data = (normalizer(alpha[1] * beta1 + (1 -alpha[1]) * beta1_fr) * torch.norm(self.fc_groups[1].weight.data, dim=-1, keepdim=True))
        self.fc_groups[2].weight.data = (normalizer(alpha[0] * beta2 + (1 - alpha[0]) * beta2_fr) * torch.norm(self.fc_groups[2].weight.data, dim=-1, keepdim=True))

        return ((beta0 * beta1).sum() + (beta2 * beta1).sum() + (beta0 * beta2).sum()) / 3

    def update_weight_four(self, alpha=0.99):

        beta0 = normalizer(self.fc_groups[0].weight.data.squeeze())
        beta1 = normalizer(self.fc_groups[1].weight.data.squeeze())
        beta2 = normalizer(self.fc_groups[2].weight.data.squeeze())
        beta3 = normalizer(self.fc_groups[3].weight.data.squeeze())

        beta0_fr =  min([((beta0 * beta1).sum().item(), 0, beta1), ((beta0 * beta2).sum().item(), 1, beta2),((beta0 * beta3).sum().item(), 2, beta3)])[2]
        beta1_fr =  min([((beta1 * beta0).sum().item(), 0, beta0), ((beta1 * beta2).sum().item(), 1, beta2),((beta1 * beta3).sum().item(), 2, beta3)])[2]
        beta2_fr =  min([((beta2 * beta0).sum().item(), 0, beta0), ((beta2 * beta1).sum().item(), 1, beta1),((beta2 * beta3).sum().item(), 2, beta3)])[2]
        beta3_fr =  min([((beta3 * beta0).sum().item(), 0, beta0), ((beta3 * beta1).sum().item(), 1, beta1),((beta3 * beta2).sum().item(), 2, beta2)])[2]

        self.fc_groups[0].weight.data = (normalizer(alpha[3] * beta0 + (1 - alpha[3]) * beta0_fr) * torch.norm(self.fc_groups[0].weight.data, dim=-1, keepdim=True))
        self.fc_groups[1].weight.data = (normalizer(alpha[2] * beta1 + (1 - alpha[2]) * beta1_fr) * torch.norm(self.fc_groups[1].weight.data, dim=-1, keepdim=True))
        self.fc_groups[2].weight.data = (normalizer(alpha[1] * beta2 + (1 - alpha[1]) * beta2_fr) * torch.norm(self.fc_groups[2].weight.data, dim=-1, keepdim=True))
        self.fc_groups[3].weight.data = (normalizer(alpha[0] * beta3 + (1 - alpha[0]) * beta3_fr) * torch.norm(self.fc_groups[3].weight.data, dim=-1, keepdim=True))

        return (
                (beta0 * beta1).sum() +
                (beta0 * beta2).sum() +
                (beta0 * beta3).sum() +
                (beta1 * beta2).sum() +
                (beta1 * beta3).sum() +
                (beta2 * beta3).sum()
            ) / 6

    def save_replay(self,dir):
        torch.save(self.real_memory_array, f'{dir}/real_replay.pt')
        torch.save(self.fake_memory_array, f'{dir}/fake_replay.pt')
        torch.save(self.real_center_array, f'{dir}/real_centers.pt')
        torch.save(self.fake_center_array, f'{dir}/fake_centers.pt')

    def CAM_forward(self, images,):
        feature_raw = self.backbone.features(images)
        feat = F.adaptive_avg_pool2d(feature_raw, (1, 1))
        feat_2d = feat.view(feat.size(0), -1)
        pred=self.test_classifier(feat_2d)
        return pred

    def update_CL(self,para,counter):
        # print_cpu_gpu_usage()
        train_data_loader=para[0]
        model=para[1]
        config=para[2]
        debug = False # (config['train_mode']=='debug')
        model.eval()
        if not debug:
            if self.config['replay_mode']=='sparse_robust':
                new_real_memory, new_fake_memory, new_real_center, new_fake_center = get_memory_v2(model,
                                                                                                train_data_loader,
                                                                                                config['alpha'],
                                                                                                config['num_pic'],
                                                                                                'cuda',
                                                                                                config=self.config)
            else:
                new_real_memory,new_fake_memory,new_real_center,new_fake_center=get_memory(model, train_data_loader, config['alpha'], config['num_pic'],'cuda',config=self.config)
        else:
            new_real_memory,new_fake_memory=torch.empty((252, 3, 256, 256)).cpu(),torch.empty((252, 3, 256, 256)).cpu()
            new_real_center,new_fake_center=torch.empty((1, 3, 256, 256)).cpu(),torch.empty((1, 3, 256, 256)).cpu()
        logger.info('---Memory Generated---')
        logger.info(f'Memory shape: real:{new_real_memory.shape}; fake:{new_fake_memory.shape}')
        self.continual_index+=1
        logger.info(f'---Current Continual Index: {self.continual_index}---')
        if counter>0:
            self.enable_teacher=True
            logger.info(f'---Teacher Enabled---')
            # self.memory.append(new_memory)
        self.real_memory_array.append(new_real_memory)
        self.fake_memory_array.append(new_fake_memory)
        self.real_center_array.append(new_real_center)
        self.fake_center_array.append(new_fake_center)
        self.real_ptr=[0 for _ in range(len(self.real_memory_array))]
        self.fake_ptr=[0 for _ in range(len(self.real_memory_array))]

        self.teacher.load_state_dict(self.backbone.state_dict())
        self.teacher_classifier=[copy.deepcopy(self.fc_groups[i]) for i in range(self.continual_index)]
        # model({'image':new_real_memory[0].unsqueeze(0).cuda()},inference=True)
        logger.info(f'---Teacher Uploaded---')
        # 冻结模型参数
        for param in self.teacher.parameters():
            param.requires_grad = False
        # 冻结模型参数
        for c in self.teacher_classifier:
            for param in c.parameters():
                param.requires_grad = False
        # print_cpu_gpu_usage()

    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        self.supcon_loss=SupConLoss()

        return loss_func

    def save_script(self):
        try:
            with open(r'training/detectors/CL_Seq_effnb4.py', 'r') as file:
                content = file.read()  # 读取文件内容
                logger.info("--------------Script Content-----------------\n%s\n-------------------------------", content)  # 打印内容
        except Exception as e:
            logger.error("Fail to read script: %s", e)  # 错误处理



    def features(self, data_dict: dict) -> torch.tensor:
        # if data_dict is not dict:
        #     tp={}
        #     tp['image']=data_dict
        #     data_dict=tp
        x = self.backbone.features(data_dict['image'])

        return x

    def classifier(self, features: torch.tensor,teacher=False,inference=False) -> torch.tensor:
        if inference:
            return self.test_classifier(features)
        else:
            return self.train_classifier(features,teacher)

    def test_classifier(self, features: torch.tensor,manner='mean') -> torch.tensor:
        sum=None
        # self.continual_index=3 this line must be remove for training, and recover for eval
        if manner == 'single-3':
            return self.fc_groups[0](features)
        if manner == 'mean':
            for index in range(self.continual_index+1):
                if sum==None:
                    sum=self.fc_groups[index](features)
                else:
                    sum+=self.fc_groups[index](features)
            return sum / (self.continual_index+1)
        if manner == 'max':
            prob_arr=[]
            for index in range(self.continual_index+1):
                prob_arr.append(self.fc_groups[index](features))
            prob_arr=torch.stack(prob_arr, dim=0)
            second_values = prob_arr[:, :, 1]
            _, max_indices = second_values.max(dim=0)
            prob_max = prob_arr[max_indices, torch.arange(max_indices.shape[0]), :]
            return prob_max
        if manner == 'min':
            prob_arr=[]
            for index in range(self.continual_index+1):
                prob_arr.append(self.fc_groups[index](features))
            prob_arr=torch.stack(prob_arr, dim=0)
            second_values = prob_arr[:, :, 1]
            _, max_indices = second_values.min(dim=0)
            prob_max = prob_arr[max_indices, torch.arange(max_indices.shape[0]), :]
            return prob_max


    def train_classifier(self, features: torch.tensor,teacher=False) -> torch.tensor:
        res_dict=[]

        start=0
        base_length=self.config['mem_each_batch']*2#self.config['train_batchSize']
        if not teacher:
            index=None
            for index in range(self.continual_index):
                res_dict.append(self.fc_groups[index](features[start:base_length]))
                start=base_length
                base_length=base_length+self.config['mem_each_batch']*2
            res_dict.append(self.fc_groups[index+1 if index is not None else 0](features[start:]))
        else:
            for index in range(self.continual_index):
                res_dict.append(self.teacher_classifier[index](features[start:base_length]))
                start=base_length
                base_length=base_length+self.config['mem_each_batch']*2
        res=torch.cat(res_dict, dim=0)

        return res

    def feature_aug(self, feat,center):
        if self.config['feat_aug_v'] == 'v1':
            return self.feature_aug_v1(feat,center)
        elif self.config['feat_aug_v'] == 'v2':
            return self.feature_aug_v2(feat,center)
        elif self.config['feat_aug_v'] == 'v3':
            return self.feature_aug_v3(feat,center)
        elif self.config['feat_aug_v'] == 'random':
            aug=random.choice([self.feature_aug_v1,self.feature_aug_v2,self.feature_aug_v3])
            return aug(feat,center)
        else:
            return feat


    def feature_aug_v1(self, feat,center):
        feat_copy = feat.clone()
        feat_mean = torch.mean(feat_copy, dim=0)

        random_index = torch.randint(0, feat_copy.size(0), (1,)).item()

        feat_copy[random_index] = feat_mean
        return feat_copy

    def feature_aug_v2(self,tensor,center):

        means = torch.zeros_like(tensor)

        means[0, :] = (tensor[0, :] + tensor[1, :]) / 2
        means[1, :] = (tensor[1, :] + tensor[2, :]) / 2
        means[2, :] = (tensor[2, :] + tensor[0, :]) / 2

        return means

    def feature_aug_v3(self,tensor,center):

        means = torch.zeros_like(tensor)
        alpha1=torch.rand(1).item()*0.5+0.5
        alpha2=torch.rand(1).item()*0.5+0.5
        alpha3=torch.rand(1).item()*0.5+0.5

        means[0, :] = tensor[0, :]*alpha1 + center*(1-alpha1)
        means[1, :] = tensor[1, :]*alpha2 + center*(1-alpha2)
        means[2, :] = tensor[2, :]*alpha3 + center*(1-alpha3)

        return means

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        new_input_size = label.shape[0]
        outputs=pred
        feature=pred_dict['feat']
        start=0
        base_length=self.config['mem_each_batch']*2#self.config['train_batchSize']
        seperate_feat=[]

        UUID_list=[]
        UUID_init=0
        loss_supcon=0
        if pred.shape[0]>new_input_size:
            # print_cpu_gpu_usage(title="loc-6")
            feature_2d = pred_dict['feat_2d']
            norm_feat = F.normalize(feature_2d)
            old_label=torch.empty((0)).cuda()
            for index in range(self.continual_index):

                if self.config.get('feat_aug',False):
                    norm_feat[start:start+self.config['mem_each_batch']]=self.feature_aug(norm_feat[start:start+self.config['mem_each_batch']],self.fake_center_array[index].cuda()) # 作为一种更强的向心约束
                    norm_feat[start+self.config['mem_each_batch']:start+2*self.config['mem_each_batch']] = self.feature_aug(
                        norm_feat[start+self.config['mem_each_batch']:start+2*self.config['mem_each_batch']], self.real_center_array[index].cuda())  # 作为一种更强的向心约束
                ones=torch.ones(self.config['mem_each_batch']).cuda()
                zeros = torch.zeros(self.config['mem_each_batch']).cuda()
                old_label=torch.cat((old_label, ones, zeros))
                UUID_list+=[UUID_init]*self.config['mem_each_batch']*2
                UUID_init+=1
                start = base_length
                base_length = base_length+self.config['mem_each_batch']*2
            UUID_list+=[UUID_init]*new_input_size # pred.shape[0]
            label=torch.cat((old_label,label),dim=0)
            data_dict['label']=label
            try:
                supcon_label = torch.tensor(UUID_list).cuda() * 10 + label
                if self.config.get('protocol','P1')=='P2':
                    for idx, l in enumerate(label):
                        if l == 0:
                            supcon_label[idx] = 0
                loss_supcon = self.supcon_loss(norm_feat, supcon_label)

            except Exception as e:
                logger.warning(e)
                logger.info(feature_2d.shape)
                logger.info(torch.tensor(UUID_list).cuda() * 10 + label)
                loss_supcon=0
            # supcon_loss(torch.cat((norm_feat.unsqueeze(1), norm_feat.unsqueeze(1)), dim=1),
            #             torch.tensor(UUID_list).cuda() * 10 + label, temperature=self.config.get('temperature', 0.1))
            # loss_supcon=supcon_loss(norm_feat.unsqueeze(1), torch.tensor(UUID_list).cuda() * 10 + label,
            #             temperature=self.config.get('temperature',0.1))
        if self.config.get('frozen', 'semi'):
            loss_ce = self.loss_func(pred, label)
        else:
            loss_ce = self.loss_func(pred[-new_input_size:], label[-new_input_size:])
        loss= loss_ce+loss_supcon*self.config.get('supcon_weight',0.1)
        loss_dict = {'overall': loss,'loss_ce':loss_ce,'loss_supcon':loss_supcon}
        if 'tea_out' in pred_dict:
            # teacher_feature = pred_dict['tea_feat']
            teacher_outputs = pred_dict['tea_out']
            old_teacher_feature = pred_dict['tea_feat']
            old_feature = feature[:-new_input_size]
            loss_fd = loss_FD(old_feature, old_teacher_feature)  # calculate FD_loss # loss_fd需要放大
            #
            loss_kd = loss_fn_kd(outputs[:-new_input_size], label[:-new_input_size], teacher_outputs, T=20.0, alpha=0.3)  # calculate KD_Loss
            if self.config.get('protocol', 'P1') == 'P1' and self.config.get('sup_task_weight',0)>0:
                loss_supcon_task=self.supcon_loss(norm_feat,torch.tensor(UUID_list).cuda())
                loss_dict['loss_sup_task'] = loss_supcon_task
                loss_dict['overall'] += loss_supcon_task*self.config.get('sup_task_weight',0.1)
            loss_dict['loss_fd']=loss_fd
            loss_dict['overall']+=loss_fd*self.config.get('fd_weight',1.0)
            loss_dict['loss_kd']=loss_kd*self.config.get('kd_weight',1.0)
            loss_dict['overall']+=loss_kd
            if torch.isnan(loss_dict['overall']).any():
                logger.info(loss_dict)
                for key,val in loss_dict.items():
                    loss_dict[key]=0
                logger.warning('datadict,preddict,pred,label')
                logger.info(data_dict)
                logger.info(pred_dict)
                logger.info(pred)
                logger.info(label)
                logger.warning('loss_overall encounter NaN')
                logger.info('')
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = torch.ceil(data_dict['label'].clamp(max=1).float()).long()
        pred = pred_dict['cls']
        if pred.shape[0]>label.shape[0]:
            ones=torch.ones((pred.shape[0]-label.shape[0])//2).cuda()
            zeros = torch.zeros((pred.shape[0] - label.shape[0]) // 2).cuda()
            label=torch.cat((label, ones,zeros))
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def generate_batch(self,memory_image, batch_size, ptr):
        batch_data = memory_image[ptr:ptr + batch_size, :, :, :]
        ptr = (ptr + batch_size) % memory_image.shape[0]  # move pointer
        return batch_data.cuda(), ptr

    def forward(self, data_dict: dict, inference=False) -> dict:
        pred_dict={}
        #label = data_dict['label']
        if self.enable_teacher and not inference:
            with torch.no_grad():
                old_group=torch.empty((0, 3, 256, 256)).cuda()
                # print_cpu_gpu_usage(title="loc-1")
                # new_image_random fake-1 real-1 fake-2 real-2
                for index in range(self.continual_index):
                    real_memory_images,ptr = self.generate_batch(self.real_memory_array[index], self.config['mem_each_batch'], self.real_ptr[index])
                    self.real_ptr[index]=ptr

                    fake_memory_images,ptr = self.generate_batch(self.fake_memory_array[index], self.config['mem_each_batch'], self.fake_ptr[index])
                    self.fake_ptr[index]=ptr
                    old_group = torch.cat((old_group,fake_memory_images,real_memory_images))
                # print_cpu_gpu_usage(title="loc-2")
                data_dict['image'] = torch.cat((old_group,data_dict['image']))
                total_len = len(data_dict['image'])
                # shuffled_img = grid_shuffle_tensor(data_dict['image'])
                # data_dict['image']=torch.cat((data_dict['image'],shuffled_img),dim=0)
                # features_teacher = self.teacher.features(
                #     torch.cat((data_dict['image'][:-total_len-data_dict['label'].shape[0]],data_dict['image'][total_len:-data_dict['label'].shape[0]])))
                features_teacher = self.teacher.features(data_dict['image'][:-data_dict['label'].shape[0]])
                feat = F.adaptive_avg_pool2d(features_teacher, (1, 1))
                features_teacher_2d = feat.view(feat.size(0), -1)
                # get the prediction by classifier
                pred_teacher = self.classifier(features_teacher_2d,inference=inference,teacher=True)
                pred_dict['tea_out'] = pred_teacher
                pred_dict['tea_feat'] = features_teacher
                # print_cpu_gpu_usage(title="loc-3")
        # get the features by backbone, all features are consistently extracted from same backbone
        features = self.features(data_dict)

        feat = F.adaptive_avg_pool2d(features, (1, 1))
        feat_2d = feat.view(feat.size(0), -1)
        # get the prediction by classifier
        pred = self.classifier(feat_2d,inference=inference)
        if inference==True:
            od_feat = F.adaptive_avg_pool2d(features, (1, 1))
            od_feat = od_feat.view(od_feat.size(0), -1)
            pred_dict['od-feat']=od_feat
        # get the probability of the pred
        # if pred_raw.size(1)>2:
        #     pred=torch.stack([pred_raw[:, 0], torch.sum(pred_raw[:, 1:], dim=1)], dim=1)
        # else:
        #     pred=pred_raw
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict.update({'cls': pred, 'prob': prob, 'feat': features, 'feat_2d':feat_2d})
        # if self.enable_teacher and not inference:
        #     # print_cpu_gpu_usage(title="loc-4")
        return pred_dict



if __name__ == '__main__':

    with open(r'D:\tencent_transfer\code\DeepfakeBench\training\config\detector\CL_LID_effnb4.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/train_config_CL.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    if config['manualSeed'] is None:
        config['manualSeed'] = 1
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])
    config['train_batchSize']=2
    detector=CL_LIDEfficientDetector(config=config).cuda()



    config['data_manner'] = 'lmdb'
    config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'
    config['sample_size']=256
    config['with_mask']=True
    config['with_landmark']=True
    config['use_data_augmentation']=True
    config['mem_each_batch']=1
    config['train_dataset'] = ['UADFV']
    config['num_pic']=12
    train_set = DeepfakeAbstractBaseDataset(config=config, mode='train')
    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_batchSize'],
            shuffle=True,
            num_workers=0,
            collate_fn=train_set.collate_fn,
        )
    optimizer = optim.Adam(
        params=detector.parameters(),
        lr=config['optimizer']['adam']['lr'],
        weight_decay=config['optimizer']['adam']['weight_decay'],
        betas=(config['optimizer']['adam']['beta1'], config['optimizer']['adam']['beta2']),
        eps=config['optimizer']['adam']['eps'],
        amsgrad=config['optimizer']['adam']['amsgrad'],
    )
    from tqdm import tqdm
    # detector.update_CL([train_data_loader,detector,config],1,debug=True)
    # detector.update_CL([train_data_loader,detector,config],1,debug=True)
    config['debug']=True
    detector.update_CL([train_data_loader,detector,config],1)

    for iteration, batch in enumerate(tqdm(train_data_loader)):
        print(iteration)
        batch['image'],batch['label'],batch['mask']=batch['image'].cuda(),batch['label'].cuda(),batch['mask'].cuda()

        # batch = {'image': torch.randn((2, 3, 256, 256), requires_grad=False).cuda(),
        #          'label': torch.randn((2), requires_grad=False).cuda(), 'mask': None}
        # flop = FlopCountAnalysis(detector, batch)
        # flop = flop.total()
        # print(f"模型的总 FLOPs: {flop / 1e6:.2f} MFLOPs")
        predictions=detector(batch)

        losses = detector.get_losses(batch, predictions)
        optimizer.zero_grad()
        losses['overall'].backward()
        optimizer.step()

        detector.post_update(epoch=10)
        with torch.no_grad():
            detector.eval()
            predictions=detector(batch,inference=True)
        if iteration > 10:
            break
