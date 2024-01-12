import numpy as np
import torch


def mDice(pred:np.array, target:np.array, epsilon = 0):
    smooth = epsilon
    m1 = pred.reshape(-1)  # Flatten
    m2 = target.reshape(-1)  # Flatten
    intersection = (m1 == m2).sum()
    # print()
    print(intersection)
    print((m1.sum() + m2.sum() + smooth))
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def mDice_Bool(mask, labelp):
    epsilon = 0.1
    labelp = (labelp==1)
    loss = ((labelp & mask).sum() + epsilon) / ((labelp | mask).sum()  + epsilon)
    return loss

def mDice_caculate(masks, target, index):
    loss_min = 1.0 * 1e6
    for mask in masks:
        mask = mask['segmentation']
        t = (target == index)
        loss = min(loss_min, mDice(mask, t, epsilon=1e-4))
    return loss


if __name__ == '__main__':

    target = np.array([[1, 2], [3, 4]])
    print(target == 1)