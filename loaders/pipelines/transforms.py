import mmcv
import torch
import numpy as np
from PIL import Image
from numpy import random
from mmdet.datasets.builder import PIPELINES
from copy import deepcopy


@PIPELINES.register_module()
class PadMultiViewImage(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, img):
        if self.size_divisor is not None:
            pad_h = int(np.ceil(img.shape[0] / self.size_divisor)) * self.size_divisor
            pad_w = int(np.ceil(img.shape[1] / self.size_divisor)) * self.size_divisor
        else:
            pad_h, pad_w = self.size

        pad_width = ((0, pad_h - img.shape[0]), (0, pad_w - img.shape[1]), (0, 0))
        img = np.pad(img, pad_width, constant_values=self.pad_val)
        return img

    def _pad_imgs(self, results):
        padded_img = [self._pad_img(img) for img in results['img']]
        
        results['ori_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_imgs(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32).reshape(-1)
        self.std = 1 / np.array(std, dtype=np.float32).reshape(-1)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        normalized_imgs = []

        for img in results['img']:
            img = img.astype(np.float32)
            if self.to_rgb:
                img = img[..., ::-1]
            img = img - self.mean
            img = img * self.std
            normalized_imgs.append(img)

        results['img'] = normalized_imgs
        results['img_norm_cfg'] = dict(
            mean=self.mean,
            std=self.std,
            to_rgb=self.to_rgb
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            ori_dtype = img.dtype
            img = img.astype(np.float32)

            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                            self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]

            new_imgs.append(img.astype(ori_dtype))

        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


@PIPELINES.register_module()
class RandomTransformImage(object):
    def __init__(self, ida_aug_conf=None, training=True):
        self.ida_aug_conf = ida_aug_conf
        self.training = training

    def __call__(self, results):
        resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
        
        aug_mats = []
        if len(results['lidar2img']) == len(results['img']):
            for i in range(len(results['img'])):
                img = Image.fromarray(np.uint8(results['img'][i]))
                
                # resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
                img, ida_mat = self.img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                )
                results['img'][i] = np.array(img).astype(np.uint8)
                results['lidar2img'][i] = ida_mat @ results['lidar2img'][i]
                results['cam_intrinsics'][i] = ida_mat @ results['cam_intrinsics'][i]
                aug_mats.append(ida_mat)

        elif len(results['img']) == 6:
            for i in range(len(results['img'])):
                img = Image.fromarray(np.uint8(results['img'][i]))
                
                # resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
                img, ida_mat = self.img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                )
                results['img'][i] = np.array(img).astype(np.uint8)

            for i in range(len(results['lidar2img'])):
                results['lidar2img'][i] = ida_mat @ results['lidar2img'][i]

        else:
            raise ValueError()

        results['ori_shape'] = [img.shape for img in results['img']]
        results['img_shape'] = [img.shape for img in results['img']]
        results['pad_shape'] = [img.shape for img in results['img']]
        results["aug_mats"] = aug_mats

        return results

    def img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        """
        https://github.com/Megvii-BaseDetection/BEVStereo/blob/master/dataset/nusc_mv_det_dataset.py#L48
        """
        def get_rot(h):
            return torch.Tensor([
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ])

        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)

        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        
        A = get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b

        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b

        ida_mat = torch.eye(4)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran

        return img, ida_mat.numpy()

    def sample_augmentation(self):
        """
        https://github.com/Megvii-BaseDetection/BEVStereo/blob/master/dataset/nusc_mv_det_dataset.py#L247
        """
        H, W = self.ida_aug_conf['H'], self.ida_aug_conf['W']
        fH, fW = self.ida_aug_conf['final_dim']

        if self.training:
            resize = np.random.uniform(*self.ida_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.ida_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.ida_aug_conf['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0

        return resize, resize_dims, crop, flip, rotate


@PIPELINES.register_module()
class GlobalRotScaleTransImage(object):
    def __init__(self,
                 rot_range=[-0.3925, 0.3925],
                 scale_ratio_range=[0.95, 1.05],
                 translation_std=[0, 0, 0]):
        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

    def __call__(self, results):
        # random rotate
        rot_angle = np.random.uniform(*self.rot_range)
        self.rotate_z(results, rot_angle)
        results["gt_bboxes_3d"].rotate(np.array(rot_angle))

        # random scale
        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        self.scale_xyz(results, scale_ratio)
        results["gt_bboxes_3d"].scale(scale_ratio)

        # TODO: support translation

        return results

    def rotate_z(self, results, rot_angle):
        rot_cos = torch.cos(torch.tensor(rot_angle))
        rot_sin = torch.sin(torch.tensor(rot_angle))

        rot_mat = torch.tensor([
            [rot_cos, -rot_sin, 0, 0],
            [rot_sin, rot_cos, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        rot_mat_inv = torch.inverse(rot_mat)

        for view in range(len(results['lidar2img'])):
            results['lidar2img'][view] = (torch.tensor(results['lidar2img'][view]).float() @ rot_mat_inv).numpy()

    def scale_xyz(self, results, scale_ratio):
        scale_mat = torch.tensor([
            [scale_ratio, 0, 0, 0],
            [0, scale_ratio, 0, 0],
            [0, 0, scale_ratio, 0],
            [0, 0, 0, 1],
        ])
        scale_mat_inv = torch.inverse(scale_mat)

        for view in range(len(results['lidar2img'])):
            results['lidar2img'][view] = (torch.tensor(results['lidar2img'][view]).float() @ scale_mat_inv).numpy()


@PIPELINES.register_module()
class PrepapreImageInputs(object):
    """Prepare the original mage inputs for the SSL.
    """
    def __init__(self, 
                 img_size, 
                 norm_cfg=None,
                 load_future_img=False,
                 load_prev_img=False,
                 **kwargs):
        self.input_size = img_size  # (h, w)
        if norm_cfg is not None:
            self.mean = norm_cfg['mean']
            self.std = norm_cfg['std']
        else:
            self.mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.std = np.array([255.0, 255.0, 255.0], dtype=np.float32)

        self.load_prev_img = load_prev_img
        self.load_future_img = load_future_img
    
    def transform_core(self, 
                       img, 
                       img_size, # (h, w)
                       to_rgb=True):
        ## we need [0, 1] images in RGB order
        img = mmcv.imresize(img, img_size[::-1])
        img = mmcv.imnormalize(np.array(img), self.mean, self.std, to_rgb)
        return img
    
    def __call__(self, results):
        img_aug = deepcopy(results['img'])

        ## resize the image
        imgs = [
            self.transform_core(img, self.input_size)
            for img in img_aug
        ]

        # process multiple imgs in single frame
        imgs = [img.transpose(2, 0, 1) for img in imgs]  # to (c, h, w)
        results['target_imgs'] = np.ascontiguousarray(np.stack(imgs, axis=0))

        results['K'] = torch.from_numpy(
                np.asarray(results['cam_intrinsics'])).to(torch.float32)
        
        ## process the intrinsic matrix
        img_shape = results['img_shape'][0]
        img_h, img_w = img_shape[0], img_shape[1]
        h, w = self.input_size[0], self.input_size[1]
        results['K'][:, 0] *= w / img_w
        results['K'][:, 1] *= h / img_h
        results['inv_K'] = torch.pinverse(results['K'])
        results['projection_mat'] = np.float32(
            np.stack(results['lidar2img']))
        return results
    

@PIPELINES.register_module()
class PointToMultiViewDepth(object):
    def __init__(self, 
                 depth_scale=(1.0, 50.0), 
                 downsample=1, 
                 render_size=(224, 352)):
        self.downsample = downsample
        self.depth_scale = depth_scale
        self.render_size = render_size

    def __call__(self, results):
        points_lidar = results['points']
        img_ori_shape = results['img_shape']

        lidarseg = None
        if 'lidarseg' in results.keys() and \
            not isinstance(results['lidarseg'], str):
            lidarseg = results['lidarseg']
            
            pts_semantic_mask = results.get('pts_semantic_mask', None)
            if pts_semantic_mask is not None:
                lidarseg = lidarseg[pts_semantic_mask]

        depth_map_list = []
        semantic_map_list = []

        for cid in range(len(img_ori_shape)):
            lidar2img = torch.from_numpy(results['lidar2img'][cid]).float()  # (4, 4)
            _img_ori_shape = img_ori_shape[cid]
            
            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)

            depth_map_raw, semantic_map = self.points2depth_and_semantic(
                points_img, _img_ori_shape, 
                self.render_size[0], self.render_size[1], lidarseg
            )
            
            depth_map_list.append(depth_map_raw)
            semantic_map_list.append(semantic_map)
            # intricics_list.append(cam2img)
            # pose_spatial_list.append(cam2camego)

        results['render_gt_depth'] = torch.stack(depth_map_list)
        results['render_gt_semantic'] = torch.stack(semantic_map_list)
        return results
    
    def points2depth_and_semantic(self, points, img_shape, height, width, semantic=None):
        rH, rW = img_shape[0], img_shape[1]
        height, width = height, width
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        semantic_map = torch.ones((height, width), dtype=torch.long) * 255
        coor = torch.round(points[:, :2] / torch.Tensor([rW/width, rH/height]))
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
                 coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                 depth < self.depth_scale[1]) & (
                 depth >= self.depth_scale[0])
        if semantic is None or type(semantic) == str:
            semantic = depth
        coor, depth, semantic = coor[kept1], depth[kept1], semantic[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks, semantic = coor[sort], depth[sort], ranks[sort], semantic[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth, semantic = coor[kept2], depth[kept2], semantic[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        semantic_map[coor[:, 1], coor[:, 0]] = semantic.long()
        return depth_map, semantic_map
