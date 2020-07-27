#oding=utf-8
import os
import datetime
import pandas as pd
from dataset.dataset_index import collate_fn1, collate_fn2, dataset
import torch
import torch.nn as nn
import torch.utils.data as torchdata
from torchvision import datasets, models, transforms
from torchvision.models import resnet50
import torch.optim as optim
from torch.optim import lr_scheduler
from utils.train_util_index import train, trainlog
from utils.warmup_scheduler import WarmupMultiStepLR
from utils.auto_resume import AutoResumer
from  torch.nn import CrossEntropyLoss
import logging
from models.resnet_index import resnet_swap_2loss_add as Extractor
from models.classifier import Classifier
from PIL import Image
import argparse
import numpy as np
import random

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', default='CUB', type=str, choices=['CUB', 'STCAR', 'AIR'])
parser.add_argument('--stage', '-s', default=3, type=int , choices=[1, 2, 3])
parser.add_argument('--num_positive', '-n', default=3, type=int)
parser.add_argument('--desc', default='_')
args = parser.parse_args()

stage_to_size = {3: '7x', 2: '14x', 1: '28x'}
num_positive = args.num_positive
cfg = {}
time = datetime.datetime.now()
# set dataset, include{CUB_200_2011: CUB, Stanford car: STCAR, JDfood: FOOD}
print("USE DATASET           <<< {} >>>".format(args.data))
sssize = stage_to_size[args.stage]
print("CALCULATE FEATURES OF <<< {} >>>".format(sssize))
cfg['dataset'] = args.data
stage = args.stage


# prepare dataset
if cfg['dataset'] == 'CUB':
    rawdata_root = './datasets/CUB/all'
    train_pd = pd.read_csv("./datasets/CUB/train.txt",sep=" ",header=None, names=['ImageName', 'label'])
    test_pd = pd.read_csv("./datasets/CUB/test.txt",sep=" ",header=None, names=['ImageName', 'label'])
    cfg['numcls'] = 200
    numimage = 6033
if cfg['dataset'] == 'STCAR':
    rawdata_root = './datasets/STCAR/all'
    train_pd = pd.read_csv("./datasets/STCAR/train.txt",sep=" ",header=None, names=['ImageName', 'label'])
    test_pd = pd.read_csv("./datasets/STCAR/test.txt",sep=" ",header=None, names=['ImageName', 'label'])
    cfg['numcls'] = 196
    numimage = 8144
if cfg['dataset'] == 'AIR':
    rawdata_root = './datasets/AIR/all'
    train_pd = pd.read_csv("./datasets/AIR/train.txt",sep=" ",header=None, names=['ImageName', 'label'])
    test_pd = pd.read_csv("./datasets/AIR/test.txt",sep=" ",header=None, names=['ImageName', 'label'])
    cfg['numcls'] = 100
    numimage = 6667
if cfg['dataset'] == 'DOG':
    rawdata_root = './datasets/st_dog/all'
    train_pd = pd.read_csv("./datasets/st_dog/train.txt",sep=" ",header=None, names=['ImageName', 'label'])
    test_pd = pd.read_csv("./datasets/st_dog/test.txt",sep=" ",header=None, names=['ImageName', 'label'])
    cfg['numcls'] = 120
    numimage = 12000
if cfg['dataset'] == 'FLW':
    rawdata_root = './datasets/flower/all'
    train_pd = pd.read_csv("./datasets/flower/train.txt",sep=" ",header=None, names=['ImageName', 'label'])
    test_pd = pd.read_csv("./datasets/flower/test.txt",sep=" ",header=None, names=['ImageName', 'label'])
    cfg['numcls'] = 102
    numimage = 2040
if cfg['dataset'] == 'BUT':
    rawdata_root = './datasets/butterfly_200/all'
    train_pd = pd.read_csv("./datasets/butterfly_200/train.txt",sep=" ",header=None, names=['ImageName', 'label'])
    test_pd = pd.read_csv("./datasets/butterfly_200/test.txt",sep=" ",header=None, names=['ImageName', 'label'])
    cfg['numcls'] = 200
    numimage = 10270

print('Dataset:',cfg['dataset'])
print('train images:', train_pd.shape)
print('test images:', test_pd.shape)
print('num classes:', cfg['numcls'])
print("********************************************************")

print('Set transform')
data_transforms = {
    'swap': transforms.Compose([
        transforms.Resize((512,512)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomCrop((448,448)),
        transforms.RandomHorizontalFlip(),
    ]),
    'swap2': None,
    'unswap': transforms.Compose([
        transforms.Resize((512,512)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomCrop((448,448)),
        transforms.RandomHorizontalFlip(),
    ]),
    'totensor': transforms.Compose([
        transforms.Resize((448,448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'None': transforms.Compose([
        transforms.Resize((512,512)),
        transforms.CenterCrop((448,448)),
    ]),

}
data_set = {}
data_set['train'] = dataset(cfg,imgroot=rawdata_root,anno_pd=train_pd, stage=stage, num_positive=num_positive,
                            unswap=data_transforms["unswap"],swap=data_transforms["None"],swap2 = data_transforms["swap2"],totensor=data_transforms["totensor"],train=True
                        )
data_set['val'] = dataset(cfg,imgroot=rawdata_root,anno_pd=test_pd, stage=stage, num_positive=num_positive,
                          unswap=data_transforms["None"],swap=data_transforms["None"],swap2 = data_transforms["swap2"],totensor=data_transforms["totensor"],train=False
                        )
dataloader = {}
dataloader['train']=torch.utils.data.DataLoader(data_set['train'], batch_size=16,
                                            shuffle=True, num_workers=16, collate_fn=collate_fn1)
dataloader['val']=torch.utils.data.DataLoader(data_set['val'], batch_size=16,
                                            shuffle=False, num_workers=16, collate_fn=collate_fn2)

print('done')
print('**********************************************')

print('Set cache dir')
filename = args.desc + '_' + str(time.month) + str(time.day) + str(time.hour) + '_' + cfg['dataset'] + '_' + sssize + '_num_' + str(num_positive)
save_dir = './net_model/' + filename
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logfile = save_dir + '/' + filename +'.log'
trainlog(logfile)
print('done')
print('*********************************************')



print('choose model and train set')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = [Extractor(stage=stage), Classifier(2048, cfg['numcls'])]
print('swap + 2 loss')

model = [e.cuda() for e in model]
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = [nn.DataParallel(e) for e in model]

base_lr = 0.001
resume = None
start_epoch = 0
if resume is not None:
    logging.info('resuming finetune from %s'%resume)
    state_dicts = torch.load(resume)
    [m.load_state_dict(d) for m, d in zip(model, state_dicts)]
    start_epoch = None

params = []
for idx, m in enumerate(model):
    for key, value in m.named_parameters():
        if not value.requires_grad:
            continue
        lr = base_lr 
        momentum = 0.9
        if isinstance(m.module, Classifier) or 'lrx' in key:
            print('[learning rate] {} is set to x10'.format(key))
            lr = base_lr * 10
        params += [{"params": [value], "lr": lr, "momentum": momentum}]
optimizer = optim.SGD(params)

criterion = CrossEntropyLoss()
scheduler = WarmupMultiStepLR(optimizer,
                              warmup_epoch = 2,
                              milestones = [60, 120, 180, 240, 300])
resumer = AutoResumer(scheduler, save_dir)

train(cfg,
      model,
      epoch_num=360,
      start_epoch=start_epoch,
      optimizer=optimizer,
      criterion=criterion,
      scheduler=scheduler,
      resumer=resumer,
      data_set=data_set,
      data_loader=dataloader,
      num_positive=num_positive,
      save_dir=save_dir)
