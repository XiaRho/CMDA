# dataset settings
"""
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
"""
# crop_size=None, after_crop_resize_size=None
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type='CityscapesICDataset',
            image_resize_size=(1024, 512),
            image_crop_size=(512, 512),
            dataset_path='/home/ubuntu/XRH/city-scapes-script-master/cityscapes_dataset/'),
            # dataset_path='/home/x1031804104/CVPR2022/HANet/cityscapes/'),
        target=dict(
            type='DSECDataset',
            dataset_txt_path='./night_dataset_warp.txt',
            crop_size=(400, 400),
            after_crop_resize_size=(512, 512),
            outputs={'warp_image', 'events_vg'}),
        source_json_root='/home/ubuntu/XRH/city-scapes-script-master/cityscapes_dataset/'),
        # source_json_root='/home/x1031804104/CVPR2022/HANet/cityscapes/'),
    val=dict(
        type='DSECDataset',
        dataset_txt_path='./night_test_dataset_warp.txt',
        outputs={'warp_image', 'events_vg', 'label', 'img_metas'}),
    test=dict(
        type='DSECDataset',
        dataset_txt_path='./night_test_dataset_warp.txt',
        outputs={'warp_image', 'events_vg', 'label', 'img_metas'}))
