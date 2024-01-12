from util import *
from data.slicer import *

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from scipy import ndimage

Prompt_Type = ['dot1', 'dot+', 'block']
Prompt_Method = {'dot1':['center', 'random'],\
                 'dot+':['center', 'random'],\
                    'block':['nearest', 'soft']}

class BTCVSet:
    def __init__(self, img_dir:str, label_dir:str) -> None:
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_list = sorted(os.listdir(img_dir))
        self.label_list = sorted(os.listdir(self.label_dir))
        # print(self.label_list)

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.img_dir, self.img_list[index]))
        label = np.load(os.path.join(self.label_dir, self.label_list[index]))
        # print(self.label_list[index])
        return img, label

    def __len__(self):
        return len(self.img_list)
    

def Prompt_Generating(img:np.ndarray, index:int=6, target = 1, dtype:str='dot1', method='center', num:int=2):
    assert index <= 13
    assert dtype in Prompt_Type
    assert method in Prompt_Method[dtype]
    # img : [512, 512]
    w, h = img.shape
    img[img!=index] = 0
    img[img==index] = target
    if dtype == 'block':
        pos = np.nonzero(img)
        posy, posx = pos
        if method == 'nearest':          
            return [np.min(posx), np.min(posy), np.max(posx), np.max(posy)], img
        elif method == 'soft':
            return [np.max(np.min(posx)-num,0), np.max(np.min(posy)-num, 0), \
                    np.min(np.max(posx)+num, w), np.min(np.max(posy)+num, h)], img
    else:     
        if dtype == 'dot1':
            if method == 'center':
                img[img==target] = 1
                dis_map = ndimage.distance_transform_edt(img)
                result = np.where(dis_map == np.max(dis_map))
                pisy, pisx = result
                target_ids = np.stack([pisx,pisy], axis=1)
                return target_ids[random.randint(0,len(target_ids)-1)], img
            elif method == 'random':
                
                return target_ids[random.randint(0,len(target_ids) - 1)], img
        elif dtype == 'dot+':
            ret = []
            if method == 'center':
                img[img==target] = 1
                dis_map = ndimage.distance_transform_edt(img)
                result = np.where(dis_map == np.max(dis_map))
                pisy, pisx = result
                target_ids = np.stack([pisx,pisy], axis=1)
                ret.append(target_ids[random.randint(0,len(target_ids)-1)])
                result = np.nonzero(img)
                pisy, pisx = result
                target_ids = np.stack([pisx,pisy], axis=1)
                for _ in range(num-1):
                    ret.append(target_ids[random.randint(0,len(target_ids) - 1)])
                # print(target_ids)
                # print(random.randint(0,len(target_ids)-1))
                return ret, img
            elif method == 'random':
                result = np.nonzero(img)
                pisy, pisx = result
                target_ids = np.stack([pisx,pisy], axis=1)
                for _ in range(num):
                    ret.append(target_ids[random.randint(0,len(target_ids) - 1)])
                return ret, img

if __name__ == '__main__':
    S = BTCVSet('/home/vision/diska4/shy/SAM/data/RawData/Training/img_slice', \
                '/home/vision/diska4/shy/SAM/data/RawData/Training/label_npy')
    # cv2.imwrite('./GT.png', S[155, 1])
    ret = Prompt_Generating(S[3][1],\
                            # dtype = 'block',\
                            # method='nearest',\
                                )
    print(ret)