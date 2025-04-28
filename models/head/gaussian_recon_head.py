'''
Copyright (c) 2025 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2025-01-15 17:13:11
Email: haimingzhang@link.cuhk.edu.cn
Description: Using the 3DGS to reconstruct the image from the GaussinFormer.
'''
import numpy as np
import torch, torch.nn as nn
from einops import rearrange, repeat

# from mmengine.registry import MODELS
from mmdet.models.builder import MODELS
from .base_head import BaseTaskHead
from ..utils_ops.utils import get_rotation_matrix
from .common.cuda_splatting import render_cuda
from .common.gaussians import build_covariance, quaternion_to_matrix
from ..gaussian_encoder.utils import GaussianPrediction


@MODELS.register_module()
class GaussianReconHead(BaseTaskHead):
    def __init__(
        self, 
        init_cfg=None,
        depth_range=None,
        render_size=None,
        apply_loss_type=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        
        if apply_loss_type == 'all':
            self.apply_loss_type = 'all'
        elif 'random' in apply_loss_type:
            self.apply_loss_type = 'random'
            self.random_apply_loss_layers = int(apply_loss_type.split('_')[1])
        elif 'fixed' in apply_loss_type:
            self.apply_loss_type = 'fixed'
            self.fixed_apply_loss_layers = [int(item) for item in apply_loss_type.split('_')[1:]]
            print(f"Supervised fixed layers: {self.fixed_apply_loss_layers}")
        else:
            raise NotImplementedError
        self.register_buffer('zero_tensor', torch.zeros(1, dtype=torch.float))
        
        self.render_h, self.render_w = render_size
        self.min_depth, self.max_depth = depth_range

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def prepare_gaussian_args(self, gaussians):
        means = gaussians.means # b, g, 3
        scales = gaussians.scales # b, g, 3
        rotations = gaussians.rotations # b, g, 4
        opacities = gaussians.semantics # b, g, c
        origi_opa = gaussians.opacities # b, g, 1
        if origi_opa.numel() == 0:
            origi_opa = torch.ones_like(opacities[..., :1], requires_grad=False)

        bs, g, _ = means.shape
        S = torch.zeros(bs, g, 3, 3, dtype=means.dtype, device=means.device)
        S[..., 0, 0] = scales[..., 0]
        S[..., 1, 1] = scales[..., 1]
        S[..., 2, 2] = scales[..., 2]
        R = get_rotation_matrix(rotations) # b, g, 3, 3
        R_new = quaternion_to_matrix(rotations)
        M = torch.matmul(S, R)
        Cov = torch.matmul(M.transpose(-1, -2), M)
        CovInv = Cov.cpu().inverse().cuda() # b, g, 3, 3

        gaussians.covariances = build_covariance(scales, rotations)

    def forward(
        self,
        representation,
        metas=None,
        **kwargs
    ):
        num_decoder = len(representation)
        if not self.training:
            apply_loss_layers = [num_decoder - 1]
        elif self.apply_loss_type == "all":
            apply_loss_layers = list(range(num_decoder))
        elif self.apply_loss_type == "random":
            if self.random_apply_loss_layers > 1:
                apply_loss_layers = np.random.choice(num_decoder - 1, self.random_apply_loss_layers - 1, False)
                apply_loss_layers = apply_loss_layers.tolist() + [num_decoder - 1]
            else:
                apply_loss_layers = [num_decoder - 1]
        elif self.apply_loss_type == 'fixed':
            apply_loss_layers = self.fixed_apply_loss_layers
        else:
            raise NotImplementedError

        intrinsics = metas['K'].to(self.zero_tensor.device)  # (bs, 6, 4, 4)
        lidar2cam = metas['lidar2cam'].to(self.zero_tensor.device)
        extrinsics = torch.inverse(
            intrinsics.new_tensor(lidar2cam)
        )  # cam2lidar

        intrinsics = intrinsics[..., :3, :3]
        # normalize the intrinsics
        intrinsics[:, :, 0] /= self.render_w
        intrinsics[:, :, 1] /= self.render_h

        prediction = []
        for idx in apply_loss_layers:
            gaussians = representation[idx]['gaussian']
            self.prepare_gaussian_args(gaussians)
            results = self.forward_render(gaussians, intrinsics, extrinsics)
            prediction.append(results)

        output = prediction[-1] # use the last one by default
        return output

    def forward_render(self,
                       gaussians,
                       intrinsics,
                       extrinsics,):
        b, v = intrinsics.shape[:2]
        device = self.zero_tensor.device

        near = torch.ones(b, v).to(device) * self.min_depth
        far = torch.ones(b, v).to(device) * self.max_depth
        background_color = torch.zeros((3), dtype=torch.float32).to(device)

        # start rendering
        render_results = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            (self.render_h, self.render_w),
            repeat(background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(gaussians.opacities, "b g c -> (b v) g c", v=v).squeeze(-1),
            scale_invariant=False,
            use_sh=True,
            feats3D=None
        )

        keys = ["render_rgb", "render_depth", "norm", "alpha", "radii", "feats"]
        render_result_dict = dict(zip(keys, render_results))

        for key, item in render_result_dict.items():
            if isinstance(item, torch.Tensor):
                render_result_dict[key] = rearrange(item, "(b v) ... -> b v ...", b=b, v=v)

        return render_result_dict
