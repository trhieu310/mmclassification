# Refer to pytorch-image-models
_base_ = [
    '../_base_/models/vit-large-p32_table.py',
    '../_base_/datasets/table_bs64_pil_resize_autoaug.py',
    '../_base_/schedules/table_bs4096_AdamW.py',
    '../_base_/default_runtime.py'
]

model = dict(backbone=dict(img_size=384))

img_norm_cfg = dict(
    mean=[227.10889537, 225.41170933, 224.90065103],
    std=[44.78552276, 46.04599916, 45.4019016], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=384, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(384, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=384),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)
