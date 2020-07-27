# coding=utf8
from __future__ import division
import pickle
import os
import torch
import torch.utils.data as data
import PIL.Image as Image
from PIL import ImageStat
class dataset(data.Dataset):
    def __init__(self, cfg, imgroot, anno_pd, stage, num_positive, unswap=None, swap=None, swap2=None, totensor=None, train=False):
        self.root_path = imgroot
        self.paths = anno_pd['ImageName'].tolist()
        self.labels = anno_pd['label'].tolist()
        self.unswap = unswap
        self.swap = swap
        self.swap2 = swap2
        self.anno_pd = anno_pd
        self.totensor = totensor
        self.cfg = cfg
        self.train = train
        self.num_positive = num_positive

    def __len__(self):
        return len(self.paths)

    def _get_2d_zeros(self, sz):
        return [[0 for i in range(sz)] for j in range(sz)]

    def _construct_coord(self, size, dir_dict):
        swap_law = self._get_2d_zeros(size)
        for idx in [i for i in range(size)]:
            for k in dir_dict[idx]:
                if k != idx:
                    swap_law[idx][k] = 1
        return swap_law

    def __getitem__(self, item):
        img_path = os.path.join(self.root_path, self.paths[item])
        img = self.pil_loader(img_path)
        crop_num = [7, 7]
        img_unswap = self.unswap(img)

        img_unswap = self.totensor(img_unswap)

        label = self.labels[item]-1

        if self.train:
            postive_images = self.fetch_positive(self.num_positive, label, self.paths[item])
            #         0               1          2
            return img_unswap, postive_images, label
        return img_unswap, label

    def fetch_positive(self, num, label, path):
        other_img_info = self.anno_pd[(self.anno_pd.label == label + 1) & (self.anno_pd.ImageName != path)]
        other_img_info = other_img_info.sample(min(num, len(other_img_info))).to_dict('records')
        other_img_path = [os.path.join(self.root_path, e['ImageName']) for e in other_img_info]
        other_img = [self.pil_loader(img) for img in other_img_path]
        other_img_unswap = [self.unswap(img) for img in other_img]
        other_img_unswap = [self.totensor(img) for img in other_img_unswap]
        return other_img_unswap

    def pil_loader(self,imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def crop_image(self, image, cropnum):
        width, high = image.size
        crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
        crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
        im_list = []
        for j in range(len(crop_y) - 1):
            for i in range(len(crop_x) - 1):
                im_list.append(image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
        return im_list


def collate_fn1(batch):
    imgs = []
    postive_imgs = []
    labels = []
    for sample in batch:
        imgs.append(sample[0])
        postive_imgs.extend(sample[1])
        labels.append(sample[2])
    return torch.stack(imgs, 0), torch.stack(postive_imgs, 0), labels

def collate_fn2(batch):
    imgs = []
    label = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])
    return torch.stack(imgs, 0), label
