# dataset settings
"""
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
"""

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type='DSECDataset',
            dataset_txt_path='./day_dataset_warp.txt',
            outputs={'warp_image', 'events_vg', '19classes'}),
        target=dict(
            type='DSECDataset',
            dataset_txt_path='./night_dataset_warp.txt',
            outputs={'warp_image', 'events_vg'}),
        source_json_root='../Night/DSEC_dataset/Day/'),
    val=dict(
        type='DSECDataset',
        dataset_txt_path='./night_test_dataset_warp.txt',
        outputs={'warp_image', 'events_vg', 'label', 'img_metas'}),
    test=dict(
        type='DSECDataset',
        dataset_txt_path='./night_test_dataset_warp.txt',
        outputs={'warp_image', 'events_vg', 'label', 'img_metas'}))
