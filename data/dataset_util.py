import os
import glob
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import cv2
corrupted_files = open('corrupted_fils', 'w')


class RainDataset(Dataset):
    def __init__(self, opt, is_eval=False, is_test=False):
        super(RainDataset, self).__init__()

        if is_test:
            self.dataset = opt.test_dataset
        elif is_eval:
            self.dataset = opt.eval_dataset
        else:
            self.dataset = opt.train_dataset
        # dataset = open(self.dataset, 'r').read().split()
        self.img_list = sorted(glob.glob(self.dataset+'/data/*'))
        self.gt_list = sorted(glob.glob(self.dataset+'/gt/*'))
   
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        gt_name = self.gt_list[idx]

        img = cv2.imread(img_name,-1)
        gt = cv2.imread(gt_name,-1)

        img = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)
        gt = cv2.resize(gt, (224,224), interpolation=cv2.INTER_AREA)
        # img = np.asarray(img).transpose((2,0,1))
        # gt = np.asarray(gt).transpose((2,0,1))

        if img.dtype == np.uint8:
            img = (img / 255.0).astype('float32')
        if gt.dtype == np.uint8:
            gt = (gt / 255.0).astype('float32')

        return [img,gt]
