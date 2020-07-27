#coding=utf8
from __future__ import division
import torch
import os,time,datetime
from torch.autograd import Variable
import logging
import numpy as np
from math import ceil
from torch.nn import L1Loss, MultiLabelSoftMarginLoss, BCELoss
from torch import nn
from .rela import calc_rela
import torch.nn.functional as F


def dt():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def trainlog(logfilepath, head='%(message)s'):
    logger = logging.getLogger('mylogger')
    logging.basicConfig(filename=logfilepath, level=logging.INFO, format=head)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(head)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def mask_to_binary(x):
    N, H, W = x.shape
    x = x.view(N, H*W)
    thresholds = torch.mean(x, dim=1, keepdim=True)
    binary_x = (x > thresholds).float()
    return binary_x.view(N, H, W)

def train(cfg,
          model,
          epoch_num,
          start_epoch,
          optimizer,
          criterion,
          scheduler,
          resumer,
          data_set,
          data_loader,
          save_dir,
          num_positive):

    def calc_bce_loss(x, y, loss_f):
        x = F.sigmoid(x)
        return loss_f(x, y)
    step = -1
    logfile = os.path.join(save_dir, 'train.log')
    stage_size = model[0].module.size

    loss_weight = {'mask': 0.1,
                   'coord': 0.1,
                  }

    for epoch in range(start_epoch,epoch_num-1):
        # train phase
        scheduler.step(epoch)
        resumer.step(model, epoch)
        model = [e.train(True) for e in model]
        extractor, classifier = model
        L = len(data_loader['train'])
        logging.info('current backbone lr: %f' % scheduler.get_lr()[0])
        for batch_cnt, data in enumerate(data_loader['train']):
            step+=1
            batch_start = time.time()
            model = [e.train(True) for e in model]
            imgs, positive_imgs, labels = data

            imgs = Variable(imgs.cuda())
            positive_imgs = Variable(positive_imgs.cuda())
            labels = Variable(torch.from_numpy(np.array(labels)).cuda())

            N = imgs.size(0)
            positive_labels = labels.view(N, 1).expand((N, num_positive)).contiguous().view(-1)
            optimizer.zero_grad()

            main_cls_feature, main_img_f, main_mask, coord_loss_1 = extractor(imgs)
            main_probs = classifier(main_cls_feature)
            cls_loss_1 = criterion(main_probs, labels)
            coord_loss_1 = torch.mean(coord_loss_1 * mask_to_binary(main_mask))
            loss_1 = cls_loss_1 + coord_loss_1 * loss_weight['coord']

            positive_cls_feature, positive_imgs_f, positive_masks, coord_loss_2 = extractor(positive_imgs)
            positive_probs = classifier(positive_cls_feature)
            cls_loss_2 = criterion(positive_probs, positive_labels)
            coord_loss_2 = torch.mean(coord_loss_2 * mask_to_binary(positive_masks))
            loss_2 = cls_loss_2 + coord_loss_2 * loss_weight['coord']

            n, c, h, w = positive_imgs_f.shape
            positive_imgs_f = positive_imgs_f.view(N, num_positive, c, h, w).transpose(0, 1).contiguous().view(N*num_positive, c, h, w)
            positive_masks = positive_masks.view(N, num_positive, h, w).transpose(0, 1).contiguous().view(N*num_positive, h, w)
            all_img_features = torch.cat((main_img_f, positive_imgs_f), dim=0)
            all_pred_masks = torch.cat((main_mask, positive_masks), dim=0)
            mask_reg_loss = calc_rela(all_img_features, all_pred_masks, num_positive).mean()

            loss = loss_1 + loss_2 + mask_reg_loss * loss_weight['mask']

            loss.backward()
            optimizer.step()

            batch_end = time.time()
            stt = '[TRAIN]: Epoch {:03d} / {:03d}    |    Batch {:03d} / {:03d}    |    Loss {:6.4f} + {:6.4f} + {:6.4f}*{:.2f} + {:6.4f}*{:.2f} + {:6.4f}*{:.2f} = {:6.4f}   |   cost {}'.format(epoch+1, epoch_num-1, batch_cnt+1, L, cls_loss_1.data.item(), cls_loss_2.data.item(), coord_loss_1.data.item(), loss_weight['coord'], coord_loss_2.data.item(), loss_weight['coord'], mask_reg_loss, loss_weight['mask'], loss.data.item(), batch_end - batch_start)
            logging.info(stt)
            with open(logfile, 'a') as f:
                f.write(stt + '\n')
        logging.info('current backbone lr: %f' % scheduler.get_lr()[0])
        # val phase
        model = [e.train(False) for e in model]

        val_loss = 0
        val_corrects1 = 0
        val_corrects2 = 0
        val_corrects3 = 0
        val_size = ceil(len(data_set['val']) / data_loader['val'].batch_size)

        t0 = time.time()
        LL = len(data_loader['val'])
        for batch_cnt_val, data_val in enumerate(data_loader['val']):
            # print data
            inputs, labels = data_val

            inputs = Variable(inputs.cuda())
            labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())
            # forward
            cls_feature = extractor(inputs)
            outputs = classifier(cls_feature)

            _, preds1 = torch.max(outputs, 1)
            # statistics
            val_loss += loss.data.item()
            # batch_corrects = torch.sum((preds == labels)).data.item()
            # val_corrects += batch_corrects
            batch_corrects1 = torch.sum((preds1 == labels)).data.item()
            val_corrects1 += batch_corrects1
            logging.info('[TEST]: Batch {:03d} / {:03d}'.format(batch_cnt_val+1, LL))
        val_loss = val_loss / val_size
        val_acc1 = 1.0 * val_corrects1 / len(data_set['val'])

        t1 = time.time()
        since = t1-t0
        logging.info('--'*30)
        logging.info('current backbone lr:%f' % scheduler.get_lr()[0])

        logging.info('%s epoch[%d]-val-loss: %.4f ||val-acc@1: %.4f ||time: %d'
                        % (dt(), epoch, val_loss, val_acc1, since))

        # save model
        save_path = os.path.join(save_dir,
                'weights-%d-%d-[%.4f].pth'%(epoch,batch_cnt,val_acc1))
        time.sleep(10)
        torch.save([e.state_dict() for e in model], save_path)
        logging.info('saved model to %s' % (save_path))
        logging.info('--' * 30)

