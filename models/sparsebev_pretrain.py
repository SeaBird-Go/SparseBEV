'''
Copyright (c) 2025 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2025-04-24 15:55:39
Email: haimingzhang@link.cuhk.edu.cn
Description: Add the GaussianFormer pretraining network in the SparseBEV framework.
'''
import os
import torch
import numpy as np
from mmcv.runner import force_fp32, auto_fp16
from mmcv.runner import get_dist_info
from mmcv.runner.fp16_utils import cast_tensor_type
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.models import builder
from .losses import OPENOCC_LOSS
from .utils import GridMask, pad_multiple, GpuPhotoMetricDistortion
from .vis_utils import visualize_elements, VisElement



@DETECTORS.register_module()
class SparseBEVPretrain(MVXTwoStageDetector):
    def __init__(self,
                 lifter=None,
                 encoder=None,
                 head=None,
                 loss_cfg=None,
                 loss_input_conversion=None,
                 data_aug=None,
                 stop_prev_grad=0,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SparseBEVPretrain, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.data_aug = data_aug
        self.stop_prev_grad = stop_prev_grad
        self.color_aug = GpuPhotoMetricDistortion()
        self.grid_mask = GridMask(ratio=0.5, prob=0.7)
        self.use_grid_mask = False

        if lifter is not None:
            self.lifter = builder.build_head(lifter)
        if encoder is not None:
            self.encoder = builder.build_head(encoder)
        if head is not None:
            self.head = builder.build_head(head)
        if loss_cfg is not None:
            self.loss_func = OPENOCC_LOSS.build(loss_cfg)
        
        self.loss_input_conversion = loss_input_conversion

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_img_feat(self, img):
        if self.use_grid_mask:
            img = self.grid_mask(img)

        img_feats = self.img_backbone(img)

        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        return img_feats

    def extract_feat(self, img, img_metas):
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        assert img.dim() == 5

        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        img = img.float()

        # move some augmentations to GPU
        if self.data_aug is not None:
            if 'img_color_aug' in self.data_aug and self.data_aug['img_color_aug'] and self.training:
                img = self.color_aug(img)

            if 'img_norm_cfg' in self.data_aug:
                img_norm_cfg = self.data_aug['img_norm_cfg']

                norm_mean = torch.tensor(img_norm_cfg['mean'], device=img.device)
                norm_std = torch.tensor(img_norm_cfg['std'], device=img.device)

                if img_norm_cfg['to_rgb']:
                    img = img[:, [2, 1, 0], :, :]  # BGR to RGB

                img = img - norm_mean.reshape(1, 3, 1, 1)
                img = img / norm_std.reshape(1, 3, 1, 1)

            for b in range(B):
                img_shape = (img.shape[2], img.shape[3], img.shape[1])
                img_metas[b]['img_shape'] = [img_shape for _ in range(N)]
                img_metas[b]['ori_shape'] = [img_shape for _ in range(N)]

            if 'img_pad_cfg' in self.data_aug:
                img_pad_cfg = self.data_aug['img_pad_cfg']
                img = pad_multiple(img, img_metas, size_divisor=img_pad_cfg['size_divisor'])

        input_shape = img.shape[-2:]
        # update real input shape of each single img
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        if self.training and self.stop_prev_grad > 0:
            H, W = input_shape
            img = img.reshape(B, -1, 6, C, H, W)

            img_grad = img[:, :self.stop_prev_grad]
            img_nograd = img[:, self.stop_prev_grad:]

            all_img_feats = [self.extract_img_feat(img_grad.reshape(-1, C, H, W))]

            with torch.no_grad():
                self.eval()
                for k in range(img_nograd.shape[1]):
                    all_img_feats.append(self.extract_img_feat(img_nograd[:, k].reshape(-1, C, H, W)))
                self.train()

            img_feats = []
            for lvl in range(len(all_img_feats[0])):
                C, H, W = all_img_feats[0][lvl].shape[1:]
                img_feat = torch.cat([feat[lvl].reshape(B, -1, 6, C, H, W) for feat in all_img_feats], dim=1)
                img_feat = img_feat.reshape(-1, C, H, W)
                img_feats.append(img_feat)
        else:
            img_feats = self.extract_img_feat(img)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)

        return losses

    @force_fp32(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def inner_forward(self,
                      points=None,
                      img_metas=None,
                      img=None,
                      out_only=False,
                      **kwargs):
        img_feats = self.extract_feat(img, img_metas)

        results = {
            'imgs': img,
            'metas': kwargs,
            'points': points,
            'ms_img_feats': img_feats,
        }
        outs = self.lifter(**results)

        results.update(outs)
        outs = self.encoder(**results)
        if out_only:
            ## here we output the image features
            outs['ms_img_feats'] = results['ms_img_feats']  # list type: [(b, n, c, h, w)]
            return outs
        results.update(outs)
        outs = self.head(**results)
        results.update(outs)
        
        return results

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img=None,
                      out_only=False,
                      **kwargs):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        results = self.inner_forward(
            points, img_metas, img, out_only=out_only, **kwargs)
        
        loss_input = {}
        for loss_input_key, loss_input_val in self.loss_input_conversion.items():
            if loss_input_val in results:
                loss_input.update({
                    loss_input_key: results[loss_input_val]})
            elif loss_input_val in kwargs:
                loss_input.update({
                    loss_input_key: kwargs[loss_input_val]})
            else:
                pass
        loss, loss_dict = self.loss_func(loss_input)
        return loss_dict

    def forward_test(self, img_metas, img=None, **kwargs):
        result_dict = self.inner_forward(
            img_metas=img_metas, img=img, **kwargs)
        
        ## visualize
        local_rank, _ = get_dist_info()
        if local_rank == 0:
            save_dir = f'outputs/SparseBEVPretrain/r50_nuimg_704x256_gs_25600_rgb_only_pretrain_overfit_wo_grid_mask_864x1600/vis'
            os.makedirs(save_dir, exist_ok=True)
            ## visualize the results
            render_rgb = result_dict['render_rgb']
            gt_img = kwargs['target_imgs']

            vis_elements_list = [
                VisElement(
                    gt_img[0],
                    type='rgb',
                    need_denormalize=False,
                ),
                VisElement(
                    render_rgb[0],
                    type='rgb',
                    need_denormalize=False,
                )
            ]
            
            if 'render_gt_depth' in kwargs.keys():
                render_depth = result_dict['render_depth'].squeeze(2)
                gt_depth = kwargs['render_gt_depth']

                print(f"render depth: min: {render_depth.min()} max: {render_depth.max()}")
                print(f"render_gt_depth depth: min: {gt_depth.min()} max: {gt_depth.max()}")

                vis_elements_list.extend(
                    [
                        VisElement(
                            render_depth[0],
                            type='depth',
                        ),
                        VisElement(
                            gt_depth[0],
                            type='depth',
                            is_sparse=True,
                        )
                    ]
                )
                
            target_size = (render_rgb.shape[-2], render_rgb.shape[-1])  # (H, W)
            visualize_elements(
                vis_elements_list,
                target_size=target_size,
                save_dir=save_dir
            )
        return [None]
