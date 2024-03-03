import argparse
import json
import os
import subprocess
import uuid
from datetime import datetime
import copy

import torch
from experiments import generate_experiment_cfgs
from mmcv import Config, get_git_hash
from tools import train


def run_command(command):
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    for line in iter(p.stdout.readline, b''):
        print(line.decode('utf-8'), end='')


def rsync(src, dst):
    rsync_cmd = f'rsync -a {src} {dst}'
    print(rsync_cmd)
    run_command(rsync_cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--exp', type=int, default=None, help='Experiment id as defined in experiment.py',)
    group.add_argument('--base_config', default=None, help='Path to config file')
    parser.add_argument('--machine', type=str, choices=['local'], default='local')
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--backbone', type=str, choices=['mit_b3', 'mit_b4', 'mit_b5', ''], default='')
    parser.add_argument('--fusion', type=str, choices=['caf', 'af', 'attf', 'attfavg', ''], default='')
    parser.add_argument('--test_mode', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=-1)
    # parser.add_argument('--confidence_type', type=str, choices=['hard', 'soft_gradual', ''], default='')
    parser.add_argument('--day_events_generate', type=str, choices=['image_change', 'gray_image', 'ic_wo_cyclegan',
                                                                    'gi_wo_cyclegan', 'events_gan', 'events_esim', ''], default='')
    parser.add_argument('--img_self_res_reg', type=str, choices=['no', 'only_isr', 'mixed', ''], default='')
    parser.add_argument('--train_size', type=str, choices=['400-400', '440-440', '512-512', ''], default='')
    parser.add_argument('--icd2en', type=str, default='')
    parser.add_argument('--events_clip_range', type=float, default=-1)
    parser.add_argument('--no_plcrop', action='store_true', default=False)
    parser.add_argument('--events_bins_5_avg_1', action='store_true', default=False)
    parser.add_argument('--feature_dist', type=float, default=-1)
    parser.add_argument('--server_type', type=str, default='1')
    parser.add_argument('--cs_isr_noise', action='store_true', default=False)
    parser.add_argument('--dz_auto_threshold', action='store_true', default=False)
    parser.add_argument('--cs_cow_mask', action='store_true', default=False)
    parser.add_argument('--high_resolution_isr', action='store_true', default=False)
    parser.add_argument('--isr_mix_aug', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--random_flare', action='store_true', default=False)
    parser.add_argument('--sky_mask', action='store_true', default=False)
    parser.add_argument('--cs_isr_data_type', type=str, choices=['day', 'new_day', ''], default='')
    parser.add_argument('--dz_isr_data_type', type=str, choices=['night', 'new_night', ''], default='')
    parser.add_argument('--deflare_aug', action='store_true', default=False)
    parser.add_argument('--isr_edge', type=float, default=-1)
    parser.add_argument('--isr_edge_class_weight', type=float, default=-1)
    parser.add_argument('--shift_3_channel', action='store_true', default=False)
    parser.add_argument('--share_decoder', action='store_true', default=False)
    parser.add_argument('--half_share_decoder', action='store_true', default=False)
    parser.add_argument('--no_share_decoder', action='store_true', default=False)
    parser.add_argument('--mixed_image_to_mixed_isr', action='store_true', default=False)
    parser.add_argument('--isr_noise_dacs_type', type=str, choices=['noise', 'noise+blur', 'blur', ''], default='')
    parser.add_argument('--source_isr_parms', type=str, default='')
    parser.add_argument('--target_isr_parms', type=str, default='')
    parser.add_argument('--dacs_isr_parms', type=str, default='')
    parser.add_argument('--image_change_range', type=int, default=-1)
    parser.add_argument('--without_events', action='store_true', default=False)
    parser.add_argument('--without_isd', action='store_true', default=False)
    parser.add_argument('--isr_no_fusion', action='store_true', default=False)
    parser.add_argument('--fusion_isr', type=str, choices=['caf', 'af', 'attf', 'attfavg', ''], default='')
    parser.add_argument('--random_choice_thres', type=str, choices=['0.25', '0.75', '0.5', 'linear', 'nlinear', '0.9-0.1',
                                                                    '0.8-0.2', '0.7-0.3', '0.6-0.4', ''], default='')
    parser.add_argument('--isd_shift_type', type=str, choices=['all', 'random', 'rightdown', ''], default='')

    parser.add_argument('--loss_weight_image', type=float, default=-1)
    parser.add_argument('--loss_weight_events', type=float, default=-1)
    parser.add_argument('--loss_weight_fusion', type=float, default=-1)
    parser.add_argument('--loss_weight_img_self_res', type=float, default=-1)
    parser.add_argument('--lambda_feature_consistency', type=float, default=-1)
    parser.add_argument('--fuse_both_ice_and_e', action='store_true', default=False)

    parser.add_argument('--root_path', type=str, default='', required=True)

    args = parser.parse_args()
    assert (args.base_config is None) != (args.exp is None), \
        'Either config or exp has to be defined.'

    GEN_CONFIG_DIR = 'configs/generated/'
    JOB_DIR = 'jobs'
    cfgs, config_files = [], []

    # Training with Predefined Config
    if args.base_config is not None:
        cfg = Config.fromfile(args.base_config)
        org_cfg_data = copy.deepcopy(cfg['data'])
        # modify cfg based on the base_config
        if args.name != '':
            cfg['name'] = args.name

        if args.backbone != '':
            if cfg['uda']['train_type'] in {'cs2dsec_image+events', 'cs2dz_image+d2n-isr', 'cs2dz_image+raw-isr',
                                            'cs2dz_image+raw-isr_no-fusion', 'cs2dsec_image+events_together'}:
                cfg['model']['pretrained'] = 'pretrained/{}.pth'.format(args.backbone)
                cfg['model']['backbone_image']['type'] = args.backbone
                cfg['model']['backbone_events']['type'] = args.backbone
            elif cfg['uda']['train_type'] in {'cs2dsec_image', 'cs2dz_image'}:
                cfg['model']['pretrained'] = 'pretrained/{}.pth'.format(args.backbone)
                cfg['model']['backbone']['type'] = args.backbone
            else:
                raise ValueError('train_type = {}'.format(cfg['uda']['train_type']))

        if args.fusion != '':
            if args.fusion == 'caf':
                cfg['model']['fusion_module']['type'] = 'ConvertAvgFusion'
            elif args.fusion == 'af':
                cfg['model']['fusion_module']['type'] = 'AverageFusion'
            elif args.fusion == 'attf':
                cfg['model']['fusion_module']['type'] = 'AttentionFusion'
            elif args.fusion == 'attfavg':
                cfg['model']['fusion_module']['type'] = 'AttentionAvgFusion'

        cfg['model']['fusion_isr_module'] = dict()
        cfg['model']['fusion_isr_module']['type'] = ''
        if args.fusion_isr != '':
            if args.fusion_isr == 'caf':
                cfg['model']['fusion_isr_module']['type'] = 'ConvertAvgFusion'
            elif args.fusion_isr == 'af':
                cfg['model']['fusion_isr_module']['type'] = 'AverageFusion'
            elif args.fusion_isr == 'attf':
                cfg['model']['fusion_isr_module']['type'] = 'AttentionFusion'
            elif args.fusion_isr == 'attfavg':
                cfg['model']['fusion_isr_module']['type'] = 'AttentionAvgFusion'
        if cfg['model']['fusion_isr_module']['type'] != '':
            cfg['uda']['isr_another_fusion'] = True
        else:
            cfg['uda']['isr_another_fusion'] = False

        assert args.share_decoder + args.half_share_decoder + args.no_share_decoder <= 1
        if args.share_decoder:
            cfg['model']['decode_head']['decoder_params']['share_decoder'] = True
            cfg['model']['decode_head']['decoder_params']['half_share_decoder'] = False
        elif args.half_share_decoder:
            cfg['model']['decode_head']['decoder_params']['share_decoder'] = False
            cfg['model']['decode_head']['decoder_params']['half_share_decoder'] = True
        elif args.no_share_decoder:
            cfg['model']['decode_head']['decoder_params']['share_decoder'] = False
            cfg['model']['decode_head']['decoder_params']['half_share_decoder'] = False
        if 'share_decoder' not in cfg['model']['decode_head']['decoder_params'].keys():
            cfg['model']['decode_head']['decoder_params']['share_decoder'] = False
        if 'half_share_decoder' not in cfg['model']['decode_head']['decoder_params'].keys():
            cfg['model']['decode_head']['decoder_params']['half_share_decoder'] = False

        if cfg['uda']['train_type'] in {'cs2dsec_image+events', 'cs2dz_image+d2n-isr',
                                        'cs2dz_image+raw-isr', 'cs2dz_image+raw-isr_split',
                                        'cs2dz_image+raw-isr_no-fusion', 'cs2dsec_image+events_together'}:
            model_cfg = {'pretrained': cfg['model']['pretrained'],
                         'backbone_image': {'type': cfg['model']['backbone_image']['type']},
                         'backbone_events': {'type': cfg['model']['backbone_events']['type']},
                         'fusion_module': {'type': cfg['model']['fusion_module']['type']},
                         'fusion_isr_module': {'type': cfg['model']['fusion_isr_module']['type']},
                         'decode_head': {'decoder_params': {'share_decoder': cfg['model']['decode_head']['decoder_params']['share_decoder'],
                                                            'half_share_decoder': cfg['model']['decode_head']['decoder_params']['half_share_decoder']}}}
        elif cfg['uda']['train_type'] in {'cs2dsec_image', 'cs2dz_image'}:
            model_cfg = {'pretrained': cfg['model']['pretrained'],
                         'backbone': {'type': cfg['model']['backbone']['type']}}
        else:
            raise ValueError('train_type = {}'.format(cfg['uda']['train_type']))

        if args.test_mode:
            cfg['name'] = 'TEST_' + cfg['name']
            cfg['uda']['debug_img_interval'] = 3
            cfg['evaluation']['interval'] = 5

        if args.batch_size != -1:
            cfg['data']['samples_per_gpu'] = args.batch_size

        '''if args.confidence_type != '':
            cfg['uda']['forward_cfg']['cal_confidence'] = True
            cfg['uda']['forward_cfg']['confidence_type'] = args.confidence_type'''

        if args.day_events_generate != '':
            if args.day_events_generate == 'image_change':
                cfg['data']['train']['source']['return_GI_or_IC'] = 'image_change'
                cfg['uda']['cyclegan_itrd2en_path'] = './pretrained/cityscapes_ICD_to_dsec_EN.pth'
            elif args.day_events_generate == 'gray_image':
                cfg['data']['train']['source']['return_GI_or_IC'] = 'gray_image'
                cfg['uda']['cyclegan_itrd2en_path'] = './pretrained/cityscapes_ID_to_dsec_EN.pth'
            elif args.day_events_generate == 'ic_wo_cyclegan':
                cfg['data']['train']['source']['return_GI_or_IC'] = 'image_change'
                cfg['uda']['cyclegan_itrd2en_path'] = ''
            elif args.day_events_generate == 'gi_wo_cyclegan':
                cfg['data']['train']['source']['return_GI_or_IC'] = 'gray_image'
                cfg['uda']['cyclegan_itrd2en_path'] = ''
            elif args.day_events_generate == 'events_gan':
                cfg['data']['train']['source']['return_GI_or_IC'] = 'events_gan'
                cfg['uda']['cyclegan_itrd2en_path'] = ''
            elif args.day_events_generate == 'events_esim':
                cfg['data']['train']['source']['return_GI_or_IC'] = 'events_esim'
                cfg['uda']['cyclegan_itrd2en_path'] = ''
        elif 'return_GI_or_IC' not in cfg['data']['train']['source'].keys():
            cfg['data']['train']['source']['return_GI_or_IC'] = 'ic_wo_cyclegan'
            cfg['uda']['cyclegan_itrd2en_path'] = ''

        if args.train_size != '':
            if args.train_size == '400-400':
                cfg['data']['train']['source']['image_crop_size'] = (400, 400)
                cfg['data']['train']['target']['crop_size'] = (400, 400)
                cfg['data']['train']['target']['after_crop_resize_size'] = (400, 400)
            elif args.train_size == '440-440':
                cfg['data']['train']['source']['image_crop_size'] = (440, 440)
                cfg['data']['train']['target']['crop_size'] = (440, 440)
                cfg['data']['train']['target']['after_crop_resize_size'] = (440, 440)
            elif args.train_size == '512-512':
                cfg['data']['train']['source']['image_crop_size'] = (512, 512)
                cfg['data']['train']['target']['crop_size'] = (440, 440)
                cfg['data']['train']['target']['after_crop_resize_size'] = (512, 512)
            else:
                raise ValueError('1')

        if args.icd2en != '':
            assert cfg['data']['train']['source']['return_GI_or_IC'] == 'image_change'
            cfg['uda']['cyclegan_itrd2en_path'] = args.icd2en

        if args.events_clip_range != -1:
            cfg['data']['train']['target']['events_clip_range'] = (args.events_clip_range, args.events_clip_range)
            cfg['data']['val']['events_clip_range'] = (args.events_clip_range, args.events_clip_range)
            cfg['data']['test']['events_clip_range'] = (args.events_clip_range, args.events_clip_range)
        else:
            cfg['data']['train']['target']['events_clip_range'] = None
            cfg['data']['val']['events_clip_range'] = None
            cfg['data']['test']['events_clip_range'] = None

        if args.no_plcrop:
            cfg['uda']['pseudo_weight_ignore_top'] = 0
            cfg['uda']['pseudo_weight_ignore_bottom'] = 0

        if args.img_self_res_reg != '':
            cfg['uda']['img_self_res_reg'] = args.img_self_res_reg

        if args.events_bins_5_avg_1:
            cfg['data']['train']['target']['events_bins_5_avg_1'] = True
            cfg['data']['val']['events_bins_5_avg_1'] = True
            cfg['data']['test']['events_bins_5_avg_1'] = True
        else:
            cfg['data']['train']['target']['events_bins_5_avg_1'] = False
            cfg['data']['val']['events_bins_5_avg_1'] = False
            cfg['data']['test']['events_bins_5_avg_1'] = False

        if args.feature_dist != -1:
            cfg['uda']['imnet_feature_dist_lambda'] = args.feature_dist
        elif 'imnet_feature_dist_lambda' not in cfg['uda'].keys():
            cfg['uda']['imnet_feature_dist_lambda'] = 0

        if args.cs_isr_noise:
            cfg['data']['train']['source']['isr_noise'] = True
        elif 'isr_noise' not in cfg['data']['train']['source'].keys():
            cfg['data']['train']['source']['isr_noise'] = False

        if args.dz_auto_threshold:
            cfg['data']['train']['target']['auto_threshold'] = True
        elif 'auto_threshold' not in cfg['data']['train']['target'].keys():
            cfg['data']['train']['target']['auto_threshold'] = False

        if args.cs_cow_mask:
            cfg['data']['train']['source']['isr_cow_mask'] = True
        elif 'isr_cow_mask' not in cfg['data']['train']['source'].keys():
            cfg['data']['train']['source']['isr_cow_mask'] = False

        if args.high_resolution_isr:
            assert cfg['data']['val']['type'] == 'DarkZurichICDataset'
            cfg['data']['train']['source']['high_resolution_isr'] = True
            cfg['data']['train']['target']['high_resolution_isr'] = True
        elif 'high_resolution_isr' not in cfg['data']['train']['source'].keys():
            cfg['data']['train']['source']['high_resolution_isr'] = False
            cfg['data']['train']['target']['high_resolution_isr'] = False

        if args.isr_mix_aug:
            cfg['uda']['isr_mix_aug'] = True
        elif 'isr_mix_aug' not in cfg['uda'].keys():
            cfg['uda']['isr_mix_aug'] = False

        if args.seed != -1:
            cfg['seed'] = args.seed

        if not args.random_flare and 'random_flare' not in cfg['data']['train']['source'].keys():
            cfg['data']['train']['source']['random_flare'] = None

        if not args.sky_mask and 'sky_mask' not in cfg['uda'].keys():
            cfg['uda']['sky_mask'] = None
        elif args.sky_mask != '':
            cfg['uda']['sky_mask'] = args.sky_mask

        if args.cs_isr_data_type == '' and 'cs_isr_data_type' not in cfg['data']['train']['source'].keys():
            cfg['data']['train']['source']['cs_isr_data_type'] = 'day'
        elif args.cs_isr_data_type != '':
            cfg['data']['train']['source']['cs_isr_data_type'] = args.cs_isr_data_type

        if args.dz_isr_data_type == '' and 'dz_isr_data_type' not in cfg['data']['train']['target'].keys():
            cfg['data']['train']['target']['dz_isr_data_type'] = 'night'
        elif args.dz_isr_data_type != '':
            cfg['data']['train']['target']['dz_isr_data_type'] = args.dz_isr_data_type

        if args.deflare_aug:
            cfg['uda']['deflare_aug'] = True
        elif 'deflare_aug' not in cfg['uda'].keys():
            cfg['uda']['deflare_aug'] = False

        if cfg['uda']['deflare_aug']:
            cfg['data']['train']['target']['outputs'] = cfg['data']['train']['target']['outputs'].union({'image_deflare', 'night_isr_deflare'})
        cfg['data']['train']['target']['outputs'] = list(cfg['data']['train']['target']['outputs'])

        if args.isr_edge != -1:
            cfg['uda']['isr_edge'] = True
            cfg['uda']['lambda_isr_features'] = args.isr_edge
        if 'isr_edge' not in cfg['uda'].keys():
            cfg['uda']['isr_edge'] = False
            cfg['uda']['lambda_isr_features'] = -1

        if args.isr_edge_class_weight != -1:
            cfg['uda']['isr_edge_class_weight'] = args.isr_edge_class_weight
        elif 'isr_edge_class_weight' not in cfg['uda'].keys():
            cfg['uda']['isr_edge_class_weight'] = -1

        if args.shift_3_channel:
            assert cfg['data']['val']['type'] == 'DarkZurichICDataset'
            cfg['data']['train']['source']['shift_3_channel'] = True
            cfg['data']['train']['target']['shift_3_channel'] = True
            cfg['uda']['shift_3_channel'] = True
        elif 'shift_3_channel' not in cfg['data']['train']['source'].keys():
            cfg['data']['train']['source']['shift_3_channel'] = False
            cfg['data']['train']['target']['shift_3_channel'] = False
            cfg['uda']['shift_3_channel'] = False

        if args.mixed_image_to_mixed_isr:
            cfg['uda']['mixed_image_to_mixed_isr'] = True
        elif 'mixed_image_to_mixed_isr' not in cfg['uda'].keys():
            cfg['uda']['mixed_image_to_mixed_isr'] = False

        if args.isr_noise_dacs_type != '':
            cfg['uda']['isr_noise_dacs_type'] = args.isr_noise_dacs_type
        elif 'isr_noise_dacs_type' not in cfg['uda'].keys():
            cfg['uda']['isr_noise_dacs_type'] = ''

        if args.image_change_range != -1:
            assert cfg['data']['val']['type'] == 'DSECDataset'
            cfg['data']['train']['target']['image_change_range'] = args.image_change_range
            cfg['data']['val']['image_change_range'] = args.image_change_range
            cfg['data']['test']['image_change_range'] = args.image_change_range
        elif 'image_change_range' not in cfg['data']['train']['target'].keys():
            cfg['data']['train']['target']['image_change_range'] = 1
            cfg['data']['val']['image_change_range'] = 1
            cfg['data']['test']['image_change_range'] = 1

        if args.without_events:
            cfg['uda']['without_events'] = True
            assert cfg['data']['val']['type'] == 'DSECDataset'
            cfg['data']['val']['outputs'] = ['label', 'img_metas', 'warp_image']
            cfg['data']['test']['outputs'] = ['label', 'img_metas', 'warp_image']
        elif 'without_events' not in cfg['uda'].keys():
            cfg['uda']['without_events'] = False

        if args.without_isd:
            cfg['uda']['without_isd'] = True
        elif 'without_isd' not in cfg['uda'].keys():
            cfg['uda']['without_isd'] = False

        if args.isr_no_fusion:
            cfg['uda']['isr_no_fusion'] = True
        elif 'isr_no_fusion' not in cfg['uda'].keys():
            cfg['uda']['isr_no_fusion'] = False

        if args.fuse_both_ice_and_e:
            cfg['uda']['fuse_both_ice_and_e'] = True
        elif 'fuse_both_ice_and_e' not in cfg['uda'].keys():
            cfg['uda']['fuse_both_ice_and_e'] = False

        if args.random_choice_thres != '':
            cfg['uda']['random_choice_thres'] = args.random_choice_thres
        elif 'random_choice_thres' not in cfg['uda'].keys():
            cfg['uda']['random_choice_thres'] = ''

        if args.isd_shift_type != '':
            cfg['data']['train']['source']['shift_type'] = args.isd_shift_type
            cfg['data']['train']['target']['shift_type'] = args.isd_shift_type
            cfg['uda']['shift_type'] = args.isd_shift_type
        if 'shift_type' not in cfg['data']['train']['source'].keys():
            cfg['data']['train']['source']['shift_type'] = 'rightdown'
        if 'shift_type' not in cfg['data']['train']['target'].keys():
            cfg['data']['train']['target']['shift_type'] = 'rightdown'
        if 'shift_type' not in cfg['uda'].keys():
            cfg['uda']['shift_type'] = 'rightdown'

        if args.shift_3_channel:
            assert cfg['data']['val']['type'] == 'DarkZurichICDataset'
            cfg['data']['train']['source']['shift_3_channel'] = True
            cfg['data']['train']['target']['shift_3_channel'] = True
            cfg['uda']['shift_3_channel'] = True
        elif 'shift_3_channel' not in cfg['data']['train']['source'].keys():
            cfg['data']['train']['source']['shift_3_channel'] = False
            cfg['data']['train']['target']['shift_3_channel'] = False
            cfg['uda']['shift_3_channel'] = False

        if args.source_isr_parms != '':
            cache = args.source_isr_parms.split('-')
            cache = [float(i) for i in cache]
            assert len(cache) == 5
            cfg['data']['train']['source']['isr_parms'] = {'val_range': (cache[0], cache[1]), '_threshold': cache[2],
                                                           '_clip_range': cache[3], 'shift_pixel': int(cache[4])}
        elif 'isr_parms' not in cfg['data']['train']['source'].keys():
            cfg['data']['train']['source']['isr_parms'] = ''

        # if args.target_isr_parms != '':
        #     cache = args.target_isr_parms.split('-')
        #     cache = [float(i) for i in cache]
        #     assert len(cache) == 5
        #     cfg['data']['train']['target']['isr_parms'] = {'val_range': (cache[0], cache[1]), '_threshold': cache[2],
        #                                                    '_clip_range': cache[3], 'shift_pixel': int(cache[4])}
        # elif 'isr_parms' not in cfg['data']['train']['target'].keys():
        #     cfg['data']['train']['target']['isr_parms'] = ''
        #
        # cfg['uda']['isr_parms'] = ''
        # if args.dacs_isr_parms != '':
        #     cache = args.dacs_isr_parms.split('-')
        #     cache = [float(i) for i in cache]
        #     assert len(cache) == 5
        #     cfg['uda']['isr_parms'] = {'val_range': (cache[0], cache[1]), '_threshold': cache[2],
        #                                '_clip_range': cache[3], 'shift_pixel': int(cache[4])}
        # elif 'isr_parms' not in cfg['uda'].keys():
        #     cfg['uda']['isr_parms'] = ''

        if args.server_type != '':
            cityscapes_dataset_path = args.root_path + 'data/cityscapes/'
            dark_zurich_dataset = args.root_path + 'data/dark_zurich/'
            cfg['data']['train']['source']['dataset_path'] = cityscapes_dataset_path
            cfg['data']['train']['source_json_root'] = cityscapes_dataset_path
            if cfg['data']['val']['type'] == 'DarkZurichICDataset':
                cfg['data']['train']['target']['dataset_path'] = dark_zurich_dataset
                cfg['data']['val']['dataset_path'] = dark_zurich_dataset
                cfg['data']['test']['dataset_path'] = dark_zurich_dataset

        if int(args.loss_weight_image) != -1:
            cfg['uda']['forward_cfg']['loss_weight']['image'] = args.loss_weight_image
        if int(args.loss_weight_events) != -1:
            cfg['uda']['forward_cfg']['loss_weight']['events'] = args.loss_weight_events
        if int(args.loss_weight_fusion) != -1:
            cfg['uda']['forward_cfg']['loss_weight']['fusion'] = args.loss_weight_fusion
        if int(args.loss_weight_img_self_res) != -1:
            cfg['uda']['forward_cfg']['loss_weight']['img_self_res'] = args.loss_weight_img_self_res

        if int(args.lambda_feature_consistency) != -1:
            cfg['uda']['lambda_feature_consistency'] = args.loss_weight_img_self_res
        else:
            cfg['uda']['lambda_feature_consistency'] = -1

        if cfg['data']['val']['type'] == 'DarkZurichICDataset':
            cfg_data = {'samples_per_gpu': cfg['data']['samples_per_gpu'],
                        'train': {'source': {'dataset_path': cfg['data']['train']['source']['dataset_path'],
                                             'isr_noise': cfg['data']['train']['source']['isr_noise'],
                                             'isr_cow_mask': cfg['data']['train']['source']['isr_cow_mask'],
                                             'high_resolution_isr': cfg['data']['train']['source']['high_resolution_isr'],
                                             'random_flare': cfg['data']['train']['source']['random_flare'],
                                             'cs_isr_data_type': cfg['data']['train']['source']['cs_isr_data_type'],
                                             'shift_3_channel': cfg['data']['train']['source']['shift_3_channel'],
                                             'isr_parms': cfg['data']['train']['source']['isr_parms'],
                                             'shift_type': cfg['data']['train']['source']['shift_type']},
                                  'target': {'dataset_path': cfg['data']['train']['target']['dataset_path'],
                                             'auto_threshold': cfg['data']['train']['target']['auto_threshold'],
                                             'high_resolution_isr': cfg['data']['train']['target']['high_resolution_isr'],
                                             'outputs': cfg['data']['train']['target']['outputs'],
                                             'shift_3_channel': cfg['data']['train']['target']['shift_3_channel'],
                                             'dz_isr_data_type': cfg['data']['train']['target']['dz_isr_data_type'],
                                             'isr_parms': cfg['data']['train']['target']['isr_parms'],
                                             'shift_type': cfg['data']['train']['target']['shift_type']},
                                  'source_json_root': cfg['data']['train']['source_json_root']},
                        'val': {'dataset_path': cfg['data']['val']['dataset_path']},
                        'test': {'dataset_path': cfg['data']['test']['dataset_path']}}
        elif cfg['data']['val']['type'] == 'DSECDataset':
            cfg_data = {'samples_per_gpu': cfg['data']['samples_per_gpu'],
                        'train': {'source': {'return_GI_or_IC': cfg['data']['train']['source']['return_GI_or_IC'],
                                             'image_crop_size': cfg['data']['train']['source']['image_crop_size'],
                                             'dataset_path': cfg['data']['train']['source']['dataset_path'],
                                             'isr_noise': cfg['data']['train']['source']['isr_noise'],
                                             'isr_cow_mask': cfg['data']['train']['source']['isr_cow_mask'],
                                             'high_resolution_isr': cfg['data']['train']['source']['high_resolution_isr'],
                                             'random_flare': cfg['data']['train']['source']['random_flare'],
                                             'cs_isr_data_type': cfg['data']['train']['source']['cs_isr_data_type'],
                                             'shift_3_channel': cfg['data']['train']['source']['shift_3_channel'],
                                             'isr_parms': cfg['data']['train']['source']['isr_parms'],
                                             'shift_type': cfg['data']['train']['source']['shift_type']},
                                  'target': {'crop_size': cfg['data']['train']['target']['crop_size'],
                                             'after_crop_resize_size': cfg['data']['train']['target']['after_crop_resize_size'],
                                             'events_clip_range': cfg['data']['train']['target']['events_clip_range'],
                                             'events_bins_5_avg_1': cfg['data']['train']['target']['events_bins_5_avg_1'],
                                             'image_change_range': cfg['data']['train']['target']['image_change_range'],
                                             'isr_parms': cfg['data']['train']['target']['isr_parms'],
                                             'shift_type': cfg['data']['train']['target']['shift_type']},
                                  'source_json_root': cfg['data']['train']['source_json_root']},
                        'val': {'events_clip_range': cfg['data']['val']['events_clip_range'],
                                'events_bins_5_avg_1': cfg['data']['val']['events_bins_5_avg_1'],
                                'image_change_range': cfg['data']['val']['image_change_range'],
                                'outputs': list(cfg['data']['val']['outputs'])},
                        'test': {'events_clip_range': cfg['data']['test']['events_clip_range'],
                                 'events_bins_5_avg_1': cfg['data']['test']['events_bins_5_avg_1'],
                                 'image_change_range': cfg['data']['val']['image_change_range'],
                                 'outputs': list(cfg['data']['val']['outputs'])}}
        else:
            cfg_data = org_cfg_data

        # Specify Name and Work Directory
        exp_name = f'{args.machine}-{cfg["exp"]}'
        unique_name = f'{datetime.now().strftime("%y%m%d_%H%M")}_' \
                      f'{cfg["name"]}_{str(uuid.uuid4())[:5]}'
        child_cfg = {
            '_base_': args.base_config.replace('configs', '../..'),
            'name': unique_name,
            'work_dir': os.path.join('work_dirs', exp_name, unique_name),
            'git_rev': get_git_hash(),
            'model': model_cfg,
            'uda': {'debug_img_interval': cfg['uda']['debug_img_interval'],
                    'cyclegan_itrd2en_path': cfg['uda']['cyclegan_itrd2en_path'],
                    # 'img_self_res_reg': cfg['uda']['img_self_res_reg'],
                    'pseudo_weight_ignore_top': cfg['uda']['pseudo_weight_ignore_top'],
                    'pseudo_weight_ignore_bottom': cfg['uda']['pseudo_weight_ignore_bottom'],
                    'imnet_feature_dist_lambda': cfg['uda']['imnet_feature_dist_lambda'],
                    'isr_mix_aug': cfg['uda']['isr_mix_aug'],
                    'sky_mask': cfg['uda']['sky_mask'],
                    'deflare_aug': cfg['uda']['deflare_aug'],
                    'isr_edge': cfg['uda']['isr_edge'],
                    'lambda_isr_features': cfg['uda']['lambda_isr_features'],
                    'isr_edge_class_weight': cfg['uda']['isr_edge_class_weight'],
                    'mixed_image_to_mixed_isr': cfg['uda']['mixed_image_to_mixed_isr'],
                    'isr_noise_dacs_type': cfg['uda']['isr_noise_dacs_type'],
                    'shift_3_channel': cfg['uda']['shift_3_channel'],
                    'isr_parms': cfg['uda']['isr_parms'],
                    'isr_no_fusion': cfg['uda']['isr_no_fusion'],
                    'lambda_feature_consistency': cfg['uda']['lambda_feature_consistency'],
                    'isr_another_fusion': cfg['uda']['isr_another_fusion'],
                    'random_choice_thres': cfg['uda']['random_choice_thres'],
                    'shift_type': cfg['uda']['shift_type'],
                    'without_events': cfg['uda']['without_events'],
                    'without_isd': cfg['uda']['without_isd'],
                    'fuse_both_ice_and_e': cfg['uda']['fuse_both_ice_and_e'],
                    'forward_cfg': {# 'cal_confidence': cfg['uda']['forward_cfg']['cal_confidence'],
                                    # 'confidence_type': cfg['uda']['forward_cfg']['confidence_type'],
                                    'loss_weight': {'image': cfg['uda']['forward_cfg']['loss_weight']['image'],
                                                    'events': cfg['uda']['forward_cfg']['loss_weight']['events'],
                                                    'fusion': cfg['uda']['forward_cfg']['loss_weight']['fusion'],
                                                    'img_self_res': cfg['uda']['forward_cfg']['loss_weight']['img_self_res']}}},
            'evaluation': {'interval': cfg['evaluation']['interval']},
            'data': cfg_data,
            'seed': cfg['seed']
        }

        if cfg['uda']['train_type'] in {'cs2dsec_image+events', 'cs2dz_image+d2n-isr'}:
            child_cfg['uda']['img_self_res_reg'] = cfg['uda']['img_self_res_reg']

        cfg_out_file = f"{GEN_CONFIG_DIR}/{exp_name}/{child_cfg['name']}.json"
        os.makedirs(os.path.dirname(cfg_out_file), exist_ok=True)
        assert not os.path.isfile(cfg_out_file)
        with open(cfg_out_file, 'w') as of:
            json.dump(child_cfg, of, indent=4)
        config_files.append(cfg_out_file)
        cfgs.append(cfg)

    # Training with Generated Configs from experiments.py
    if args.exp is not None:
        exp_name = f'{args.machine}-exp{args.exp}'
        cfgs = generate_experiment_cfgs(args.exp)
        # Generate Configs
        for i, cfg in enumerate(cfgs):
            if args.debug:
                cfg.setdefault('log_config', {})['interval'] = 10
                cfg['evaluation'] = dict(interval=200, metric='mIoU')
                if 'dacs' in cfg['name']:
                    cfg.setdefault('uda', {})['debug_img_interval'] = 10
                    # cfg.setdefault('uda', {})['print_grad_magnitude'] = True
            # Generate Config File
            cfg['name'] = f'{datetime.now().strftime("%y%m%d_%H%M")}_' \
                          f'{cfg["name"]}_{str(uuid.uuid4())[:5]}'
            cfg['work_dir'] = os.path.join('work_dirs', exp_name, cfg['name'])
            cfg['git_rev'] = get_git_hash()
            cfg['_base_'] = ['../../' + e for e in cfg['_base_']]
            cfg_out_file = f"{GEN_CONFIG_DIR}/{exp_name}/{cfg['name']}.json"
            os.makedirs(os.path.dirname(cfg_out_file), exist_ok=True)
            assert not os.path.isfile(cfg_out_file)
            with open(cfg_out_file, 'w') as of:
                json.dump(cfg, of, indent=4)
            config_files.append(cfg_out_file)

    if args.machine == 'local':
        for i, cfg in enumerate(cfgs):  # 一个程序运行多个cfg
            print('Run job {}'.format(cfg['name']))
            train.main([config_files[i]])
            torch.cuda.empty_cache()
    else:
        raise NotImplementedError(args.machine)

    train_command = 'python my_run_experiments.py --base_config configs/fusion/cs2dz_image+raw-isr_b5.py --server_type ecust --name cs2dz_image+raw-isr_newday_shift3_israug_b5 --isr_mix_aug --shift_3_channel --cs_isr_data_type new_day'
    '''
    DSEC SOTA[56.89-60.05]: 
    CUDA_VISIBLE_DEVICES=2 nohup
    python my_run_experiments.py --server_type ren --base_config configs/fusion/cs2dsec_image+events_together_b5.py 
    --name cs2dsec_image+events_together_A01B0005C1_FuseBoth*_b5 --target_isr_parms 0.01-1.01-0.005-0.1-1 
    --dacs_isr_parms 0.01-1.01-0.005-0.1-1 --fuse_both_ice_and_e > ./work_dirs/cs2dsec_image+events_together_A01B0005C1_FuseBoth*_b5.out 2>&1 &
    '''
