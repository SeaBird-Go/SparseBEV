'''
Copyright (c) 2025 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2025-04-28 11:12:55
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
dataset_type = 'CustomNuScenesDatasetOverfit'
dataset_root = 'data/nuscenes/'

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True
)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

# arch config
embed_dims = 256
num_layers = 6
num_query = 900
num_frames = 8
num_levels = 4
num_points = 4

img_backbone = dict(
    type='ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN2d', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    with_cp=True)
img_neck = dict(
    type='FPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=embed_dims,
    num_outs=num_levels)
img_norm_cfg = dict(
    mean=[123.675, 116.280, 103.530],
    std=[58.395, 57.120, 57.375],
    to_rgb=True)


# ========= model config ===============
num_decoder = 4
num_single_frame_decoder = 1
pc_range = point_cloud_range
scale_range = [0.08, 0.64]
xyz_coordinate = 'cartesian'
phi_activation = 'sigmoid'
include_opa = True
include_color = True
semantics = True
semantic_dim = 17
num_groups = 4
use_deformable_func = True  # setup.py needs to be executed

render_size = (128, 352)  # (h, w)

model = dict(
    type='SparseBEVPretrain',
    data_aug=dict(
        img_color_aug=True,  # Move some augmentations to GPU
        img_norm_cfg=img_norm_cfg,
        img_pad_cfg=dict(size_divisor=32)),
    stop_prev_grad=0,
    img_backbone=img_backbone,
    img_neck=img_neck,
    lifter=dict(
        type='GaussianLifter',
        num_anchor=25600,
        embed_dims=embed_dims,
        anchor_grad=True,
        feat_grad=False,
        phi_activation=phi_activation,
        semantics=semantics,
        semantic_dim=semantic_dim,
        include_opa=include_opa,
        include_color=include_color,
    ),
    encoder=dict(
        type='GaussianOccEncoder',
        anchor_encoder=dict(
            type='SparseGaussian3DEncoder',
            embed_dims=embed_dims, 
            include_opa=include_opa,
            include_color=include_color,
            semantics=semantics,
            semantic_dim=semantic_dim
        ),
        norm_layer=dict(type="LN", normalized_shape=embed_dims),
        ffn=dict(
            type="AsymmetricFFN",
            in_channels=embed_dims * 2,
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
        ),
        deformable_model=dict(
            type='DeformableFeatureAggregation',
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=num_levels,
            num_cams=6,
            attn_drop=0.15,
            use_deformable_func=use_deformable_func,
            use_camera_embed=True,
            residual_mode="cat",
            kps_generator=dict(
                type="SparseGaussian3DKeyPointsGenerator",
                embed_dims=embed_dims,
                phi_activation=phi_activation,
                xyz_coordinate=xyz_coordinate,
                num_learnable_pts=2,
                fix_scale=[
                    [0, 0, 0],
                    [0.45, 0, 0],
                    [-0.45, 0, 0],
                    [0, 0.45, 0],
                    [0, -0.45, 0],
                    [0, 0, 0.45],
                    [0, 0, -0.45],
                ],
                pc_range=pc_range,
                scale_range=scale_range
            ),
        ),
        refine_layer=dict(
            type='SparseGaussian3DRefinementModule',
            embed_dims=embed_dims,
            pc_range=pc_range,
            scale_range=scale_range,
            restrict_xyz=True,
            unit_xyz=[4.0, 4.0, 1.0],
            refine_manual=[0, 1, 2],
            phi_activation=phi_activation,
            semantics=semantics,
            semantic_dim=semantic_dim,
            include_opa=include_opa,
            include_color=include_color,
            xyz_coordinate=xyz_coordinate,
            semantics_activation='softplus',
        ),
        spconv_layer=dict(
            # _delete_=True,
            type="SparseConv3D",
            in_channels=embed_dims,
            embed_channels=embed_dims,
            pc_range=pc_range,
            grid_size=[0.5, 0.5, 0.5],
            phi_activation=phi_activation,
            xyz_coordinate=xyz_coordinate,
            use_out_proj=True,
        ),
        num_decoder=num_decoder,
        num_single_frame_decoder=num_single_frame_decoder,
        operation_order=[
            "deformable",
            "ffn",
            "norm",
            "refine",
        ] * num_single_frame_decoder + [
            "spconv",
            "norm",
            "deformable",
            "ffn",
            "norm",
            "refine",
        ] * (num_decoder - num_single_frame_decoder),
    ),
    head=dict(
        type='GaussianReconHead',
        apply_loss_type='random_1',
        render_size=render_size,
        depth_range=[0.1, 64.0],
    ),
    # ============= loss config ================
    loss_cfg = dict(
        type='MultiLoss',
        loss_cfgs=[
            # dict(
            #     type='PhotometricLoss',
            #     weight=1.0,
            #     input_dict=dict(
            #         pred_rgb='render_rgb',
            #         gt_rgb='target_imgs')
            # ),
            dict(
                type='DepthLoss',
                weight=0.1,
                input_dict=dict(
                    pred_depth='render_depth',
                    gt_depth='render_gt_depth')
            )
        ]
    ),
    loss_input_conversion = dict(
        render_rgb="render_rgb",
        render_depth="render_depth",
        render_gt_depth="render_gt_depth"
    )
)

ida_aug_conf = {
    # 'resize_lim': (0.38, 0.55),
    'resize_lim': (0.44, 0.44),
    'final_dim': (256, 704),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900, 'W': 1600,
    'rand_flip': False,
}

train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
    ),
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    # dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1),
    # dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectNameFilter', classes=class_names),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(type="PrepapreImageInputs", img_size=render_size),
    dict(type="PointToMultiViewDepth",
         render_size=render_size
    ),
    # dict(type='GlobalRotScaleTransImage', rot_range=[-0.3925, 0.3925], scale_ratio_range=[0.95, 1.05]),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', 
         keys=['img', 'K', 'inv_K', 'target_imgs', 
               'render_gt_depth', 'projection_mat', 'lidar2cam'], 
         meta_keys=(
        'filename', 'ori_shape', 'img_shape', 'pad_shape', 
        'lidar2img', 'img_timestamp',))
]

test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
    ),
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(type="PrepapreImageInputs", img_size=render_size),
    dict(type="PointToMultiViewDepth",
         render_size=render_size
    ),
    # dict(type='GlobalRotScaleTransImage', rot_range=[-0.3925, 0.3925], scale_ratio_range=[0.95, 1.05]),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', 
         keys=['img', 'K', 'inv_K', 'target_imgs', 
               'render_gt_depth', 'projection_mat', 'lidar2cam'], 
         meta_keys=(
        'filename', 'ori_shape', 'img_shape', 'pad_shape', 
        'lidar2img', 'img_timestamp',))
]

data = dict(
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        load_interval=4,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_train_sweep.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        filter_empty_gt=False,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type + 'Val',
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_train_sweep.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    # test=dict(
    #     type=dataset_type,
    #     data_root=dataset_root,
    #     ann_file=dataset_root + 'nuscenes_infos_test_sweep.pkl',
    #     pipeline=test_pipeline,
    #     classes=class_names,
    #     modality=input_modality,
    #     test_mode=True,
    #     box_type_3d='LiDAR')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),
        'sampling_offset': dict(lr_mult=0.1),
    }),
    weight_decay=0.01
)

optimizer_config = dict(
    type='Fp16OptimizerHook',
    loss_scale=512.0,
    grad_clip=dict(max_norm=35, norm_type=2)
)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)
total_epochs = 24
batch_size = 4

# load pretrained weights
load_from = 'pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'
revise_keys = [('backbone', 'img_backbone')]

# resume the last training
resume_from = None

# checkpointing
checkpoint_config = dict(interval=1, max_keep_ckpts=3)

# logging
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', interval=10, reset_flag=True),
        # dict(type='MyTensorboardLoggerHook', interval=500, reset_flag=True)
    ]
)

# evaluation
# eval_config = dict(interval=total_epochs,
#                    dynamic_intervals=[(22, 1)])

eval_config = dict(interval=1)

# other flags
debug = False
