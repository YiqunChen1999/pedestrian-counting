# # Custom imports
# custom_imports = dict(
#     imports=['counter.data.datasets'],
#     allow_failed_imports=False)
# dataset settings
dataset_type = 'PedestrianDataset'
data_root = 'data/TJU-Ped-campus/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', img_scale=(2048, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        img_scale=(2048, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'dhd_pedestrian/ped_campus/annotations/dhd_pedestrian_campus_train.json',
        img_prefix=data_root + 'dhd_campus_train_images/dhd_campus/images/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'dhd_pedestrian/ped_campus/annotations/dhd_pedestrian_campus_val.json',
        img_prefix=data_root + 'dhd_campus/images/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'dhd_pedestrian/ped_campus/annotations/dhd_pedestrian_campus_val.json',
        img_prefix=data_root + 'dhd_campus/images/val',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric=['bbox', 'miss_rate'], proposal_nums=(1, 10, 100))
