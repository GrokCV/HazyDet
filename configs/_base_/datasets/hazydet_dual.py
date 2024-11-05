# dataset settings
dataset_type = 'HazyDetDataset'
data_root = '/opt/data/private/fcf/Public_dataset/HazyDet-365K/data/HazyDet365K/'

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        imdecode_backend='pillow',
        type='LoadDualAnnotations',
        with_bbox=True,
        with_seg=True,
        max_depth_path=data_root+'Depth/max_depth.txt'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(pad_val=dict(img=0, seg=255), size=(
        1333,
        800,
    ), type='Pad'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/train_coco.json',
        data_prefix=dict(img='train/hazy_images/',
                         seg='Depth/normalized_depth_maps/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val/val_coco.json',
        data_prefix=dict(img='val/hazy_images/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))




# inference on test dataset and
# format the output results for submission.
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test/test_coco.json',
        data_prefix=dict(img='test/hazy_images/'),
        test_mode=True,
        pipeline=test_pipeline))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val/val_coco.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)


test_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    format_only=False,
    ann_file=data_root + 'test/test_coco.json',
    backend_args=backend_args)