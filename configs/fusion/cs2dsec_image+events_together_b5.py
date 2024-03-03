_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/daformer_sepaspp_mitb5.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/uda_cityscapes_day_to_dsec_night_512x512.py',
    # Basic UDA Self-Training
    '../_base_/uda/dacs.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 0

# Modifications to Basic model
pretrained_type = 'mit_b5'
events_bins = 1
train_type = 'cs2dsec_image+events_together'
model = dict(
    type='FusionEncoderDecoder',
    pretrained='pretrained/{}.pth'.format(pretrained_type),
    backbone_image=dict(type=pretrained_type, style='pytorch', in_chans=3),
    backbone_events=dict(type=pretrained_type, style='pytorch', in_chans=3),
    fusion_module=dict(type='AttentionAvgFusion'),
    fusion_isr_module=dict(type='AttentionFusion'),
    decode_head=dict(type='DAFormerHeadFusion',
                     decoder_params=dict(train_type=train_type,
                                         share_decoder=True)),
    train_type=train_type
)

# Modifications to Basic UDA
uda = dict(
    # Increased Alpha
    alpha=0.999,
    cyclegan_itrd2en_path='./pretrained/cityscapes_ICD_to_dsec_EN.pth',
    img_self_res_reg='no',  # no, only_isr, mixed
    train_type=train_type,
    forward_cfg=dict(loss_weight={'image': 0.5, 'events': 0.5, 'fusion': 0.5, 'img_self_res': 0.25},
                     gradual_rate=0.0),
    mixed_image_to_mixed_isr=True,
    random_choice_thres='0.5',
    shift_type='random',
    isr_parms=dict(val_range=[0.01, 1.01],
                   _threshold=0.005,
                   _clip_range=0.1,
                   shift_pixel=1),
    # Thing-Class Feature Distance
    imnet_feature_dist_lambda=0,  # not use imnet_feature_dist (raw 0.005)
    imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
    imnet_feature_dist_scale_min_ratio=0.75,
    # Pseudo-Label Crop
    pseudo_weight_ignore_top=0,  # top=15, bottom=120
    pseudo_weight_ignore_bottom=0, debug_img_interval=500)  # ,debug_img_interval=10

data = dict(
    train=dict(
        # Rare Class Sampling
        rare_class_sampling=dict(min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5),
        source=dict(outputs={'image', 'img_time_res', 'img_self_res', 'label'},
                    return_GI_or_IC='image_change',
                    shift_type='random'),
        target=dict(events_bins=events_bins,
                    isr_type='real_time',
                    shift_type='random',
                    isr_parms=dict(val_range=[0.01, 1.01],
                                   _threshold=0.005,
                                   _clip_range=0.1,
                                   shift_pixel=1),
                    outputs={'warp_image', 'events_vg', 'warp_img_self_res'})),
    val=dict(events_bins=events_bins),
    test=dict(events_bins=events_bins))

# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))

n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=40000, max_keep_ckpts=1)
evaluation = dict(interval=4000, metric='mIoU')  # 4000
# Meta Information for Result Analysis

name = 'cs2dsec_image+events_b5'
exp = 'basic'
name_dataset = 'cityscapes_day2dsec_night'
name_architecture = 'daformer_sepaspp_mitb5_events'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp_events'
name_uda = 'dacs_a999_rcs0.01_cpl'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
