_base_ = [
    'pipelines/auto_aug.py',
]

# dataset settings
# dataset_type = 'ImageNet'
dataset_type = 'Table'
classes = ["FIGURE", "BBT", "LOGO", "BORDER", "NATURAL", "TABLE"]
data_dir = "../../input/illustration-classification/"
img_norm_cfg = dict(
    mean=[227.10889537, 225.41170933, 224.90065103],
    std=[44.78552276, 46.04599916, 45.4019016], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=384,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='AutoAugment', policies={{_base_.policy_imagenet}}),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        size=(384, -1),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=384),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix=data_dir + 'train',
        ann_file=data_dir + 'anns/train.txt',
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=data_dir + 'valid',
        ann_file=data_dir + 'anns/valid.txt',
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix=data_dir + 'test',
        ann_file=data_dir + 'anns/test.txt',
        classes=classes,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric=['accuracy', 'f1_score', 'recall', 'precision', 'loss'])
