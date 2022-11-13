# dataset settings
dataset_type = 'Table'
classes = ["FIGURE", "BBT", "LOGO", "BORDER", "NATURAL", "TABLE"]
data_dir = "../../input/illustration-classification/"
img_norm_cfg = dict(
    mean=[227.10889537, 225.41170933, 224.90065103], std=[44.78552276, 46.04599916, 45.4019016], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResized', size=256, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=1,
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
        classes = classes,
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix=data_dir + 'test',
        ann_file=data_dir + 'anns/test.txt',
        classes = classes,
        pipeline=test_pipeline))
evaluation = dict(interval=5, metric='accuracy')
