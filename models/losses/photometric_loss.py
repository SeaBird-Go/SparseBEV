'''
Copyright (c) 2025 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2025-01-17 10:44:25
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OPENOCC_LOSS
from .base_loss import BaseLoss


@OPENOCC_LOSS.register_module()
class PhotometricLoss(BaseLoss):
    def __init__(
        self,
        weight=1.0,
        input_dict=None
    ):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'pred_rgb': 'pred_rgb',
                'gt_rgb': 'gt_rgb',
            }
        else:
            self.input_dict = input_dict

        self.loss_func = self.loss_rgb

    def loss_rgb(self, pred_rgb, gt_rgb):
        loss = nn.functional.l1_loss(pred_rgb, gt_rgb)
        return loss
    

@OPENOCC_LOSS.register_module()
class DepthLoss(BaseLoss):
    def __init__(
        self,
        weight=1.0,
        input_dict=None,
        name='loss_depth'
    ):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'pred_depth': 'pred_depth',
                'gt_depth': 'gt_depth',
            }
        else:
            self.input_dict = input_dict

        self.loss_func = self.loss_depth
        self.loss_name = name
    
    def loss_depth(self, pred_depth, gt_depth):
        pred_depth = pred_depth.squeeze(2)

        device = pred_depth.device

        mask = gt_depth > 0.0
        loss_depth = F.l1_loss(pred_depth[mask], gt_depth[mask])
        if torch.isnan(loss_depth):
            print('NaN in render depth loss!')
            loss_depth = torch.Tensor([0.0]).to(device)
        return loss_depth
