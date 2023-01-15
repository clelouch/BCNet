import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def build_loss(config):
    loss_dict = {
        'bce': nn.BCELoss(size_average=True),
        'f3': structure_loss
        # 'hard': HardMining(config.n, config.gamma)
    }
    return loss_dict[config.mask_loss], loss_dict[config.edge_loss]
