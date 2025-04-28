'''
Copyright (c) 2025 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2025-04-01 16:22:40
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import knn
# from mmdet.models.utils import multi_apply
from mmdet.core import multi_apply
from mmdet.models.losses import SmoothL1Loss

from . import OPENOCC_LOSS
from .base_loss import BaseLoss


@OPENOCC_LOSS.register_module()
class ChamferDistanceLoss(BaseLoss):
    """Calculate the L1 chamfer distance loss between predicted pts and gt pts.
    """
    def __init__(
        self,
        weight=1.0,
        input_dict=None,
        empty_dist_thr=0.2,
        empty_weights=5.0
    ):
        super().__init__(weight)

        self.empty_dist_thr = empty_dist_thr
        self.empty_weights = empty_weights

        if input_dict is None:
            self.input_dict = {
                'gt_pts': 'gt_pts',
                'pred_pts': 'pred_pts',
            }
        else:
            self.input_dict = input_dict

        self.loss_func = self.loss_cd

        self.loss_pts = SmoothL1Loss(beta=0.2, loss_weight=0.5)

    def loss_cd(self, pred_pts, gt_pts):
        num_dec_layers = len(pred_pts)
        all_gt_points_list = [gt_pts for _ in range(num_dec_layers)]
        all_refine_pts = pred_pts

        ## Loop the layers
        losses_pts = multi_apply(
            self.loss_single, all_refine_pts, all_gt_points_list)[0]
        
        loss = 0.0
        for _loss_pts in losses_pts:
            loss = loss + _loss_pts

        loss = loss / num_dec_layers
        return loss
    
    def loss_single(self,
                    pred_pts,
                    gt_points):
        bs = gt_points.size(0)
        refine_pts = pred_pts['gaussian'].means
        refine_pts = refine_pts.reshape(bs, -1, 3)
        refine_pts_list = [refine_pts[i] for i in range(bs)]
        gt_points_list = [gt_points[i] for i in range(bs)]

        ## Loop the batch size
        (gt_paired_idx_list, pred_paired_idx_list,
         gt_pts_weights) = multi_apply(
             self._get_target_single, refine_pts_list, gt_points_list)
        
        gt_paired_pts, pred_paired_pts= [], []
        for i in range(bs):
            gt_paired_pts.append(refine_pts_list[i][gt_paired_idx_list[i]])
            pred_paired_pts.append(gt_points_list[i][pred_paired_idx_list[i]])
        
        # concatenate all results from different samples
        gt_pts = torch.cat(gt_points_list)
        gt_paired_pts = torch.cat(gt_paired_pts)
        gt_pts_weights = torch.cat(gt_pts_weights)
        pred_pts = torch.cat(refine_pts_list)
        pred_paired_pts = torch.cat(pred_paired_pts)

        # calculate loss pts
        loss_pts = pred_pts.new_tensor(0)
        loss_pts += self.loss_pts(gt_pts,
                                  gt_paired_pts,
                                  weight=gt_pts_weights[..., None],
                                  avg_factor=gt_pts.shape[0])
        loss_pts += self.loss_pts(pred_pts, 
                                  pred_paired_pts,
                                  avg_factor=pred_pts.shape[0])

        loss_tmp = pred_pts.new_tensor(0)
        return (loss_pts,)
        
    
    @torch.no_grad()
    def _get_target_single(self, refine_pts, gt_points):
        # knn to apply Chamfer distance
        gt_paired_idx = knn(1, refine_pts[None, ...], gt_points[None, ...])
        gt_paired_idx = gt_paired_idx.permute(0, 2, 1).squeeze().long()
        pred_paired_idx = knn(1, gt_points[None, ...], refine_pts[None, ...])
        pred_paired_idx = pred_paired_idx.permute(0, 2, 1).squeeze().long()
        gt_paired_pts = refine_pts[gt_paired_idx]
        # pred_paired_pts = gt_points[pred_paired_idx]

        # gt side assignment
        gt_pts_weights = refine_pts.new_ones(gt_paired_pts.shape[0])
        dist = torch.norm(gt_points - gt_paired_pts, dim=-1)
        mask = (dist > self.empty_dist_thr)
        gt_pts_weights[mask] = self.empty_weights

        return (gt_paired_idx, pred_paired_idx, gt_pts_weights)