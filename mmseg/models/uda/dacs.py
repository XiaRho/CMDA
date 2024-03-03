# The ema model and the domain-mixing are based on:
# https://github.com/vikolss/DACS

import math
import os
import random
from copy import deepcopy
from PIL import Image

import mmcv.runner.hooks.logger.text
import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.cyclegan import define_G, LightNet
from mmseg.models.uda.uda_decorator import UDADecoratorEvents, UDADecoratorFusion, UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks, get_mean_std, strong_transform,
                                                sky_mask_transform, seg_label_to_edge_label, add_noise_on_isr)
from mmseg.models.utils.visualization import subplotimg
from mmseg.utils.utils import downscale_label_ratio
from mmseg.datasets.utils import get_image_change_from_pil
from mmseg.models.uda.prototype_contrast import ContrastCELoss

plt.switch_backend('agg')


def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@UDA.register_module()
class DACS(UDADecoratorFusion):

    def __init__(self, **cfg):
        super(DACS, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.debug_img_interval = cfg['debug_img_interval']
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

        self.train_type = cfg['train_type']
        assert self.train_type in {'cs2dsec_image', 'cs2dsec_image+events', 'cs2dz_image', 'cs2dz_image+d2n-isr',
                                   'cs2dz_image+raw-isr', 'cs2dz_image+raw-isr_no-fusion', 'cs2dz_image+raw-isr_split',
                                   'cs2dsec_image+events_together'}

        self.forward_cfg = cfg['forward_cfg']
        if 'img_self_res_reg' in cfg.keys():
            self.img_self_res_reg = cfg['img_self_res_reg']
            assert self.img_self_res_reg in {'no', 'only_isr', 'mixed', 'average'}

        self.isr_mix_aug = False
        if 'isr_mix_aug' in cfg.keys() and cfg['isr_mix_aug']:
            self.isr_mix_aug = True

        if 'cyclegan_itrd2en_path' in cfg.keys() and cfg['cyclegan_itrd2en_path'] != '' and \
                self.train_type in {'cs2dsec_image+events', 'cs2dsec_image+events_together'}:
            self.cyclegan_itrd2en = define_G().cuda()
            cyclegan_model_pth = torch.load(cfg['cyclegan_itrd2en_path'])
            self.cyclegan_itrd2en.load_state_dict(cyclegan_model_pth)
            self.cyclegan_itrd2en.eval()
        else:
            self.cyclegan_itrd2en = None

        if 'cyclegan_id2in_path' in cfg.keys() and cfg['cyclegan_id2in_path'] != '' and self.train_type == 'cs2dz_image':
            self.cyclegan_id2in = define_G(input_nc=3, output_nc=3).cuda()
            cyclegan_model_pth = torch.load(cfg['cyclegan_id2in_path'])
            self.cyclegan_id2in.load_state_dict(cyclegan_model_pth)
            self.cyclegan_id2in.eval()
            self.mean_torch = torch.tensor([[[[0.485]], [[0.456]], [[0.406]]]]).cuda()
            self.std_torch = torch.tensor([[[[0.229]], [[0.224]], [[0.225]]]]).cuda()
        else:
            self.cyclegan_id2in = None

        if 'cyclegan_light_path' in cfg.keys() and cfg['cyclegan_light_path'] != '' and self.train_type == 'cs2dz_image':
            self.cyclegan_light = LightNet().cuda()
            cyclegan_model_pth = torch.load(cfg['cyclegan_light_path'])
            self.cyclegan_light.load_state_dict(cyclegan_model_pth)
            self.cyclegan_light.eval()
            self.mean_torch = torch.tensor([[[[0.485]], [[0.456]], [[0.406]]]]).cuda()
            self.std_torch = torch.tensor([[[[0.229]], [[0.224]], [[0.225]]]]).cuda()
        else:
            self.cyclegan_light = None

        self.sky_mask = cfg['sky_mask']
        if self.sky_mask is not None:
            self.sky_mask_parm = dict()
            self.sky_mask_parm['noise_root_path'] = self.sky_mask
            self.sky_mask_parm['noise_list'] = os.listdir(self.sky_mask)

        self.deflare_aug = False
        if 'deflare_aug' in cfg.keys() and cfg['deflare_aug']:
            self.deflare_aug = True
            assert self.train_type in {'cs2dz_image+raw-isr'}

        self.isr_edge = False
        if 'isr_edge' in cfg.keys() and cfg['isr_edge']:
            self.isr_edge = True
            assert self.train_type in {'cs2dz_image+raw-isr_no-fusion'}
            self.lambda_isr_features = cfg['lambda_isr_features']

        self.isr_edge_class_weight = None
        if 'isr_edge_class_weight' in cfg.keys() and cfg['isr_edge_class_weight'] != -1:
            assert 0 <= cfg['isr_edge_class_weight'] <= 1
            self.isr_edge_class_weight = [1 - cfg['isr_edge_class_weight'], cfg['isr_edge_class_weight']]
            self.get_model().decode_head.loss_decode.class_weight = self.isr_edge_class_weight

        self.mixed_image_to_mixed_isr = False
        if 'mixed_image_to_mixed_isr' in cfg.keys() and cfg['mixed_image_to_mixed_isr']:
            self.mixed_image_to_mixed_isr = True
            self.mixed_image_to_mixed_isr_parms = {'val_range': (1, 10 ** 2), '_threshold': 0.04, '_clip_range': 0.2, 'shift_pixel': 3}

        self.isr_noise_dacs_type = ''
        if 'isr_noise_dacs_type' in cfg.keys() and cfg['isr_noise_dacs_type'] != '':
            assert self.mixed_image_to_mixed_isr
            self.isr_noise_dacs_type = cfg['isr_noise_dacs_type']
            assert self.isr_noise_dacs_type in {'noise', 'noise+blur', 'blur'}

        self.shift_3_channel = False
        if 'shift_3_channel' in cfg.keys() and cfg['shift_3_channel']:
            self.shift_3_channel = True
            self.mixed_image_to_mixed_isr_parms = [
                {'val_range': (9, 255 + 9), '_threshold': 0.012, '_clip_range': 0.04, 'shift_pixel': 1},
                {'val_range': (9, 255 + 9), '_threshold': 0.012, '_clip_range': 0.12, 'shift_pixel': 3},
                {'val_range': (9, 255 + 9), '_threshold': 0.012, '_clip_range': 0.20, 'shift_pixel': 5}]

        if 'isr_parms' in cfg.keys() and cfg['isr_parms'] != '':
            assert not self.shift_3_channel
            assert isinstance(cfg['isr_parms'], dict)
            self.mixed_image_to_mixed_isr_parms = cfg['isr_parms']

        if 'without_events' in cfg.keys() and cfg['without_events']:
            assert self.train_type == 'cs2dsec_image+events'
            self.without_events = True
        else:
            self.without_events = False

        if 'without_isd' in cfg.keys() and cfg['without_isd']:
            assert self.train_type == 'cs2dsec_image+events'
            self.without_isd = True
        else:
            self.without_isd = False

        assert not (self.without_events and self.without_isd)

        if 'isr_no_fusion' in cfg.keys() and cfg['isr_no_fusion']:
            assert self.train_type == 'cs2dsec_image+events'
            self.isr_no_fusion = True
        else:
            self.isr_no_fusion = False

        if 'lambda_feature_consistency' in cfg.keys() and cfg['lambda_feature_consistency'] != -1:
            self.forward_cfg['lambda_feature_consistency'] = cfg['lambda_feature_consistency']
        else:
            self.forward_cfg['lambda_feature_consistency'] = 0.25

        if 'isr_another_fusion' in cfg.keys() and cfg['isr_another_fusion']:
            assert self.train_type in {'cs2dsec_image+events', 'cs2dsec_image+events_together'}
            self.isr_another_fusion = True
        else:
            self.isr_another_fusion = False

        self.events_isr_choice_start_thres = -1
        self.events_isr_choice_end_thres = -1
        if 'random_choice_thres' in cfg.keys() and cfg['random_choice_thres'] != '':
            assert cfg['random_choice_thres'] in {'0.25', '0.75', '0.5', 'linear', 'nlinear',
                                                  '0.9-0.1', '0.8-0.2', '0.7-0.3', '0.6-0.4'}
            if cfg['random_choice_thres'] in {'0.25', '0.75', '0.5'}:
                self.random_choice_thres = float(cfg['random_choice_thres'])
            elif '-' in cfg['random_choice_thres']:
                assert len(cfg['random_choice_thres'].split('-')) == 2
                self.events_isr_choice_start_thres = float(cfg['random_choice_thres'].split('-')[0])
                self.events_isr_choice_end_thres = float(cfg['random_choice_thres'].split('-')[1])
            elif cfg['random_choice_thres'] == 'linear':
                self.events_isr_choice_start_thres = 1.0
                self.events_isr_choice_end_thres = 0.0
            elif cfg['random_choice_thres'] == 'nlinear':
                self.events_isr_choice_start_thres = 0.0
                self.events_isr_choice_end_thres = 1.0
        else:
            self.random_choice_thres = 0.5

        if 'shift_type' in cfg.keys() and cfg['shift_type'] != '':
            self.shift_type = cfg['shift_type']
        else:
            self.shift_type = 'rightdown'
        assert self.shift_type in {'all', 'random', 'rightdown'}

        self.fuse_both_ice_and_e = False
        if 'fuse_both_ice_and_e' in cfg.keys() and cfg['fuse_both_ice_and_e']:
            self.fuse_both_ice_and_e = True
            assert self.train_type == 'cs2dsec_image+events_together'

        if self.enable_fdist:
            cfg_imnet = deepcopy(cfg['model'])
            if self.train_type in {'cs2dsec_image+events', 'cs2dz_image+d2n-isr',
                                   'cs2dz_image+raw-isr', 'cs2dsec_image+events_together'}:
                cfg_imnet['type'] = 'EncoderDecoder'
                cfg_imnet['backbone'] = cfg_imnet['backbone_image']
            self.imnet_model = build_segmentor(cfg_imnet)
        else:
            self.imnet_model = None

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        if 'target_img_metas' in data_batch.keys():
            log_vars.pop('loss', None)  # remove the unnecessary 'loss'
            outputs = dict(log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        else:
            log_vars.pop('loss', None)  # remove the unnecessary 'loss'
            if 'image' in data_batch['source'].keys():
                num_samples = data_batch['source']['image'].shape[0]
            else:
                num_samples = data_batch['target']['warp_image'].shape[0]
            outputs = dict(log_vars=log_vars, num_samples=num_samples)
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        with torch.no_grad():
            self.get_imnet_model().eval()
            feat_imnet = self.get_imnet_model().extract_feat(img)
            '''if isinstance(feat_imnet, dict):
                feat_imnet = feat_imnet['f_image']'''
            feat_imnet = [f.detach() for f in feat_imnet]
        lay = -1
        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                self.fdist_scale_min_ratio,
                                                self.num_classes,
                                                255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                              fdist_mask)
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescale = gt_rescaled
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def forward_train(self, **kwargs):

        ################################################################
        ################### load source and target data
        ################################################################
        day_events, night_events = None, None
        if self.train_type in {'cs2dsec_image', 'cs2dz_image'}:
            night_key = 'warp_image' if 'warp_image' in kwargs['target'].keys() else 'image'
            day_image = kwargs['source']['image']
            day_label = kwargs['source']['label']
            night_image = kwargs['target'][night_key]
            if self.train_type == 'cs2dz_image' and self.cyclegan_id2in is not None:
                with torch.no_grad():
                    day_image = (day_image * self.std_torch + self.mean_torch - 0.5) / 0.5
                    day_image = self.cyclegan_id2in(day_image)
                    day_image = (day_image / 2 + 0.5 - self.mean_torch) / self.std_torch
            elif self.train_type == 'cs2dz_image' and self.cyclegan_light is not None:
                if self.local_iter == 0:
                    self.get_model().cyclegan_light = self.cyclegan_light
                # with torch.no_grad():
                #     day_image += self.cyclegan_light(day_image)
                #     night_image += self.cyclegan_light(night_image)
        elif self.train_type == 'cs2dz_image+d2n-isr':
            day_image = kwargs['source']['image']
            day_label = kwargs['source']['label']
            night_image = kwargs['target']['image']
            night_isr = kwargs['target']['night_isr']
            target_day_image = kwargs['target']['day_image']
            target_day_t_isr = kwargs['target']['day_t_isr']
        elif self.train_type in {'cs2dz_image+raw-isr', 'cs2dz_image+raw-isr_split', 'cs2dz_image+raw-isr_no-fusion'}:
            day_image = kwargs['source']['image']
            day_isr = kwargs['source']['img_self_res']
            day_label = kwargs['source']['label']
            if 'warp_image' in kwargs['target'].keys():  # cs2dsec
                night_image = kwargs['target']['warp_image']
                night_isr = kwargs['target']['warp_img_self_res']
            else:
                night_image = kwargs['target']['image']
                night_isr = kwargs['target']['night_isr']
        else:
            assert self.train_type in {'cs2dsec_image+events', 'cs2dsec_image+events_together'}
            day_image = kwargs['source']['image']
            day_isr = kwargs['source']['img_self_res']
            if self.cyclegan_itrd2en is not None:
                with torch.no_grad():
                    kwargs['source']['img_time_res'] = torch.mean(kwargs['source']['img_time_res'], dim=1, keepdim=True)
                    day_events = self.cyclegan_itrd2en(kwargs['source']['img_time_res'])
                    day_events = day_events.repeat(1, 3, 1, 1)
            else:
                day_events = kwargs['source']['img_time_res']

            day_label = kwargs['source']['label']
            night_image = kwargs['target']['warp_image']
            night_events = kwargs['target']['events_vg']
            night_isr = kwargs['target']['warp_img_self_res']
            if self.without_events:
                self.forward_cfg['isr_events_fusion_choice'] = -1
            elif self.without_isd:
                self.forward_cfg['isr_events_fusion_choice'] = 2
            else:
                self.forward_cfg['isr_events_fusion_choice'] = torch.rand(1).detach()

            if self.events_isr_choice_start_thres != -1 and self.events_isr_choice_end_thres != -1:
                self.random_choice_thres = self.events_isr_choice_start_thres + (self.events_isr_choice_end_thres
                                           - self.events_isr_choice_start_thres) * self.local_iter / self.max_iters

        log_vars = {}
        batch_size = day_image.shape[0] if day_image is not None else day_events.shape[0]
        dev = day_image.device if day_image is not None else day_events.device

        if self.deflare_aug:
            night_image_deflare = kwargs['target']['image_deflare']
            night_isr_deflare = kwargs['target']['night_isr_deflare']

        if self.sky_mask is not None:
            with torch.no_grad():
                for i in range(batch_size):
                    day_isr[i] = sky_mask_transform(param=self.sky_mask_parm, isr=day_isr[i], label=day_label[i])

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        means, stds = get_mean_std(None, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0),
            'sigma': random.uniform(0.15, 1.15)
        }

        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        ################################################################
        ################### Train on source date
        ################################################################
        if self.train_type in {'cs2dsec_image', 'cs2dz_image'}:
            source_ce_losses, pred = self.get_model().forward_train(day_image, day_events, day_label, return_feat=True)
        elif self.train_type == 'cs2dz_image+d2n-isr':
            with torch.no_grad():
                target_day_out = self.get_ema_model().encode_decode(img=target_day_image, events=None, output_features=False,
                                                                    test_cfg={'output_type': 'image'})  # [1, 19, H, W] fusion_output
                ema_target_day_softmax = torch.softmax(target_day_out, dim=1)
                target_day_pl_prob, target_day_pl = torch.max(ema_target_day_softmax, dim=1)

            inputs = {'image': day_image, 'events': target_day_t_isr}
            target_day_pl = target_day_pl[:, None]
            source_label = {'image': day_label, 'events': target_day_pl}
            source_ce_losses, pred = self.get_model().forward_train(inputs, source_label, return_feat=True,
                                                                    cfg=self.forward_cfg)
        elif self.train_type in {'cs2dz_image+raw-isr', 'cs2dz_image+raw-isr_split', 'cs2dz_image+raw-isr_no-fusion'}:
            inputs = {'image': day_image, 'events': day_isr}
            if self.train_type == 'cs2dz_image+raw-isr_no-fusion' and self.isr_edge:
                source_label = {'image': day_label, 'events': seg_label_to_edge_label(day_label)}
            else:
                source_label = day_label
            source_ce_losses, pred = self.get_model().forward_train(inputs, source_label, return_feat=True,
                                                                    cfg=self.forward_cfg)
        elif self.train_type == 'cs2dsec_image+events_together':
            source_label = day_label
            inputs = {'image': day_image, 'events': day_events, 'img_self_res': day_isr}
            if self.fuse_both_ice_and_e:
                source_ce_losses, pred = self.get_model().forward_train(inputs, source_label, return_feat=True,
                                                                        cfg=dict(self.forward_cfg, **{'fusion_all': True}))
            elif self.isr_another_fusion and not (self.forward_cfg['isr_events_fusion_choice'] > self.random_choice_thres):  # isr
                source_ce_losses, pred = self.get_model().forward_train(inputs, source_label, return_feat=True,
                                                                        cfg=dict(self.forward_cfg, **{'fusion_isr': True}))
            else:  # events
                source_ce_losses, pred = self.get_model().forward_train(inputs, source_label, return_feat=True,
                                                                        cfg=self.forward_cfg)
        else:
            assert self.train_type == 'cs2dsec_image+events'
            source_label = day_label
            inputs = {'image': day_image}
            if self.forward_cfg['isr_events_fusion_choice'] > self.random_choice_thres:  # events
                inputs['events'] = day_events
            else:  # isr
                inputs['events'] = day_isr
            if self.isr_no_fusion and not (self.forward_cfg['isr_events_fusion_choice'] > self.random_choice_thres):  # isr
                source_ce_losses, pred = self.get_model().forward_train(inputs, source_label, return_feat=True,
                                                                        cfg=dict(self.forward_cfg, **{'no_fusion': True}))
            elif self.isr_another_fusion and not (self.forward_cfg['isr_events_fusion_choice'] > self.random_choice_thres):
                source_ce_losses, pred = self.get_model().forward_train(inputs, source_label, return_feat=True,
                                                                        cfg=dict(self.forward_cfg, **{'fusion_isr': True}))
            else:
                source_ce_losses, pred = self.get_model().forward_train(inputs, source_label, return_feat=True,
                                                                        cfg=self.forward_cfg)

        src_feat = source_ce_losses.pop('features')
        source_ce_loss, clean_log_vars = self._parse_losses(source_ce_losses)  # ['decode.loss_seg', 'decode.acc_seg']
        log_vars.update(clean_log_vars)
        source_loss = source_ce_loss
        source_loss.backward(retain_graph=self.enable_fdist)

        ################################################################
        ################### create source data visualization
        ################################################################
        with torch.no_grad():
            if self.train_type in {'cs2dsec_image', 'cs2dz_image'}:
                day_img_softmax = torch.softmax(pred, dim=1)  # img
                _, day_img_seg = torch.max(day_img_softmax, dim=1)
            elif self.train_type == 'cs2dz_image+d2n-isr':
                day_img_softmax = torch.softmax(pred['image_output'], dim=1)  # img
                _, day_img_seg = torch.max(day_img_softmax, dim=1)
                day_events_softmax = torch.softmax(pred['events_output'], dim=1)  # events
                _, day_events_seg = torch.max(day_events_softmax, dim=1)
            elif self.train_type in {'cs2dz_image+raw-isr', 'cs2dz_image+raw-isr_split', 'cs2dz_image+raw-isr_no-fusion'}:
                assert self.img_self_res_reg == 'no'
                day_img_softmax = torch.softmax(pred['image_output'], dim=1)  # img
                _, day_img_seg = torch.max(day_img_softmax, dim=1)
                day_events_softmax = torch.softmax(pred['events_output'], dim=1)  # events
                _, day_events_seg = torch.max(day_events_softmax, dim=1)
                if self.train_type == 'cs2dz_image+raw-isr':
                    day_fusion_softmax = torch.softmax(pred['fusion_output'], dim=1)  # fusion
                    _, day_fusion_seg = torch.max(day_fusion_softmax, dim=1)
            else:
                assert self.train_type in {'cs2dsec_image+events', 'cs2dsec_image+events_together'}
                if self.train_type == 'cs2dsec_image+events_together':
                    day_isr_softmax = torch.softmax(pred['img_self_res_output'], dim=1)  # img_self_res
                    _, day_isr_seg = torch.max(day_isr_softmax, dim=1)
                day_img_softmax = torch.softmax(pred['image_output'], dim=1)  # img
                _, day_img_seg = torch.max(day_img_softmax, dim=1)
                day_events_softmax = torch.softmax(pred['events_output'], dim=1)  # events
                _, day_events_seg = torch.max(day_events_softmax, dim=1)
                if not self.isr_no_fusion or self.forward_cfg['isr_events_fusion_choice'] > self.random_choice_thres:  # events
                    day_fusion_softmax = torch.softmax(pred['fusion_output'], dim=1)  # fusion
                    _, day_fusion_seg = torch.max(day_fusion_softmax, dim=1)

        if self.print_grad_magnitude:  # False
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # ImageNet feature distance
        if self.enable_fdist:
            if self.train_type in {'cs2dsec_image', 'cs2dz_image'}:
                feat_loss, feat_log = self.calc_feat_dist(day_image, day_label, src_feat)
            elif self.train_type in {'cs2dz_image+d2n-isr', 'cs2dz_image+raw-isr'}:
                feat_loss, feat_log = self.calc_feat_dist(day_image, day_label, src_feat['f_image'])
            else:
                assert self.train_type in {'cs2dsec_image+events', 'cs2dsec_image+events_together'}
                feat_loss, feat_log = self.calc_feat_dist(day_image, day_label, src_feat['f_image'])
            feat_loss.backward()
            log_vars.update(add_prefix(feat_log, 'src'))
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')

        ################################################################
        ################### create target PL
        ################################################################
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        with torch.no_grad():
            if self.train_type in {'cs2dsec_image', 'cs2dz_image'}:
                ema_logits = self.get_ema_model().encode_decode(night_image, None)
                ema_img_softmax = torch.softmax(ema_logits, dim=1)  # img
                pseudo_prob, pseudo_label = torch.max(ema_img_softmax, dim=1)
                ema_img_seg = pseudo_label
            elif self.train_type == 'cs2dz_image+d2n-isr':
                ema_logits = self.get_ema_model().encode_decode(img=night_image, events=night_isr, output_features=True)
                ema_img_softmax = torch.softmax(ema_logits['image_output'], dim=1)  # img
                _, ema_img_seg = torch.max(ema_img_softmax, dim=1)
                ema_events_softmax = torch.softmax(ema_logits['events_output'], dim=1)  # events
                _, ema_events_seg = torch.max(ema_events_softmax, dim=1)
                assert self.img_self_res_reg == 'average'
                gradual_pseudo_softmax = 0.5 * ema_img_softmax + 0.5 * ema_events_softmax
                pseudo_prob, pseudo_label = torch.max(gradual_pseudo_softmax, dim=1)
            elif self.train_type in {'cs2dz_image+raw-isr', 'cs2dz_image+raw-isr_split', 'cs2dz_image+raw-isr_no-fusion'}:
                if self.deflare_aug:
                    ema_logits = self.get_ema_model().encode_decode(night_image_deflare, night_isr_deflare, output_features=True)
                else:
                    ema_logits = self.get_ema_model().encode_decode(night_image, night_isr, output_features=True)
                ema_img_softmax = torch.softmax(ema_logits['image_output'], dim=1)  # img
                _, ema_img_seg = torch.max(ema_img_softmax, dim=1)
                ema_events_softmax = torch.softmax(ema_logits['events_output'], dim=1)  # events
                _, ema_events_seg = torch.max(ema_events_softmax, dim=1)
                if self.train_type == 'cs2dz_image+raw-isr':
                    ema_fusion_softmax = torch.softmax(ema_logits['fusion_output'].detach(), dim=1)  # fusion
                    pseudo_prob_f, pseudo_label_f = torch.max(ema_fusion_softmax, dim=1)
                    pseudo_prob, pseudo_label = pseudo_prob_f, pseudo_label_f
                elif self.train_type == 'cs2dz_image+raw-isr_no-fusion':
                    # utilize image results to supervise isr results, thus the image_output is pseudo_label
                    pseudo_prob, pseudo_label = torch.max(ema_img_softmax, dim=1)
                else:
                    pseudo_prob_image, pseudo_label_image = torch.max(ema_img_softmax.detach(), dim=1)
                    pseudo_prob_events, pseudo_label_events = torch.max(ema_events_softmax.detach(), dim=1)

                    ps_large_p_image = pseudo_prob_image.ge(self.pseudo_threshold).long() == 1  # > 0.968
                    ps_size = np.size(np.array(pseudo_label_image.cpu()))
                    pseudo_weight_image = torch.sum(ps_large_p_image).item() / ps_size
                    pseudo_weight_image = pseudo_weight_image * torch.ones(pseudo_prob_image.shape, device=dev)

                    ps_large_p_events = pseudo_prob_events.ge(self.pseudo_threshold).long() == 1  # > 0.968
                    ps_size = np.size(np.array(pseudo_label_events.cpu()))
                    pseudo_weight_events = torch.sum(ps_large_p_events).item() / ps_size
                    pseudo_weight_events = pseudo_weight_events * torch.ones(pseudo_prob_events.shape, device=dev)

                    if self.psweight_ignore_top > 0:
                        pseudo_weight_image[:, :self.psweight_ignore_top, :] = 0
                    if self.psweight_ignore_bottom > 0:
                        pseudo_weight_image[:, -self.psweight_ignore_bottom:, :] = 0

                    if self.psweight_ignore_top > 0:
                        pseudo_weight_events[:, :self.psweight_ignore_top, :] = 0
                    if self.psweight_ignore_bottom > 0:
                        pseudo_weight_events[:, -self.psweight_ignore_bottom:, :] = 0

                    gt_pixel_weight = torch.ones(pseudo_weight_events.shape, device=dev)
            else:
                assert self.train_type in {'cs2dsec_image+events', 'cs2dsec_image+events_together'}
                if self.forward_cfg['isr_events_fusion_choice'] > self.random_choice_thres:  # events
                    ema_imputs_events_isr = night_events
                else:  # isr
                    ema_imputs_events_isr = night_isr

                if self.fuse_both_ice_and_e:
                    ema_logits = self.get_ema_model().encode_decode(night_image, night_events,
                                                                    img_self_res=night_isr,
                                                                    output_features=True,
                                                                    test_cfg=dict(self.forward_cfg, **{'fusion_all': True}))
                elif self.isr_another_fusion and not (self.forward_cfg['isr_events_fusion_choice'] > self.random_choice_thres):
                    ema_logits = self.get_ema_model().encode_decode(night_image, night_isr, output_features=True,
                                                                    test_cfg=dict(self.forward_cfg, **{'fusion_isr': True}))
                elif self.isr_no_fusion:
                    ema_logits = self.get_ema_model().encode_decode(night_image, night_events, output_features=True,
                                                                    test_cfg=self.forward_cfg)
                else:
                    ema_logits = self.get_ema_model().encode_decode(night_image, ema_imputs_events_isr, output_features=True,
                                                                    test_cfg=self.forward_cfg)

                ema_img_softmax = torch.softmax(ema_logits['image_output'], dim=1)  # img
                _, ema_img_seg = torch.max(ema_img_softmax, dim=1)
                ema_events_softmax = torch.softmax(ema_logits['events_output'], dim=1)  # events
                _, ema_events_seg = torch.max(ema_events_softmax, dim=1)
                # ema_isr_softmax = torch.softmax(ema_logits['img_self_res_output'], dim=1)  # events
                # _, ema_isr_seg = torch.max(ema_isr_softmax, dim=1)
                ema_fusion_softmax = torch.softmax(ema_logits['fusion_output'].detach(), dim=1)  # fusion
                pseudo_prob_f, pseudo_label_f = torch.max(ema_fusion_softmax, dim=1)
                pseudo_prob, pseudo_label = pseudo_prob_f, pseudo_label_f

                '''if self.img_self_res_reg == 'mixed':
                    gradual_rate = self.local_iter / self.max_iters  # increased values
                    if 'gradual_rate' in self.forward_cfg.keys():
                        self.forward_cfg['gradual_rate'] = gradual_rate
                    gradual_pseudo_softmax = gradual_rate * ema_fusion_softmax + (1 - gradual_rate) * \
                                             ema_img_self_res_softmax
                    pseudo_prob, pseudo_label = torch.max(gradual_pseudo_softmax, dim=1)
                elif self.img_self_res_reg == 'average':
                    gradual_pseudo_softmax = 0.5 * ema_fusion_softmax + 0.5 * ema_img_self_res_softmax
                    pseudo_prob, pseudo_label = torch.max(gradual_pseudo_softmax, dim=1)
                elif self.img_self_res_reg == 'only_isr':
                    pseudo_prob, pseudo_label = pseudo_prob_isr, pseudo_label_isr
                elif self.img_self_res_reg == 'no':
                    pseudo_prob, pseudo_label = pseudo_prob_f, pseudo_label_f
                else:
                    raise ValueError('error self.img_self_res_reg = {}'.format(self.img_self_res_reg))'''

        if self.train_type != 'cs2dz_image+raw-isr_split':
            ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1  # > 0.968
            ps_size = np.size(np.array(pseudo_label.cpu()))
            pseudo_weight = torch.sum(ps_large_p).item() / ps_size
            pseudo_weight = pseudo_weight * torch.ones(pseudo_prob.shape, device=dev)

            if self.psweight_ignore_top > 0:
                pseudo_weight[:, :self.psweight_ignore_top, :] = 0
            if self.psweight_ignore_bottom > 0:
                pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
            gt_pixel_weight = torch.ones(pseudo_weight.shape, device=dev)

        ################################################################
        ################### Mix source and target data
        ################################################################
        mixed_img, mixed_lbl, mixed_events, mixed_isr = [None] * batch_size, [None] * batch_size, \
                                                        [None] * batch_size, [None] * batch_size
        mixed_lbl_2 = [None] * batch_size
        mix_masks = get_class_masks(day_label)  # 0(target) or 1(source)

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            if day_image is not None:
                mixed_img[i], _ = strong_transform(strong_parameters, data=torch.stack((day_image[i], night_image[i])))
            if day_events is not None:
                _, mixed_events[i] = strong_transform(strong_parameters, target=torch.stack((day_events[i], night_events[i])))
            if self.train_type in {'cs2dz_image+raw-isr', 'cs2dz_image+raw-isr_split', 'cs2dz_image+raw-isr_no-fusion',
                                   'cs2dsec_image+events', 'cs2dsec_image+events_together'}:
                if self.mixed_image_to_mixed_isr:
                    mixed_i_np = torch.clamp(denorm(mixed_img[i], means, stds), 0, 1) * 255
                    mixed_i_np = np.transpose(mixed_i_np.cpu().numpy()[0], (1, 2, 0))
                    # mixed_i_np = (np.transpose(mixed_img[i][0].cpu().numpy(), (1, 2, 0)) + 1) * 127.5
                    mixed_i_pil = Image.fromarray(np.uint8(mixed_i_np))
                    if not self.shift_3_channel:
                        if self.shift_type == 'random':
                            direct = [['leftdown', 'leftup'], ['rightdown', 'rightup']]
                            this_shift_direction = direct[int(strong_parameters['color_jitter'] * 10)
                                                          % 2][int(strong_parameters['color_jitter'] * 100) % 2]
                        else:
                            this_shift_direction = self.shift_type
                        mixed_isr[i] = get_image_change_from_pil(mixed_i_pil, width=512, height=512, auto_threshold=None,
                                                                 shift_direction=this_shift_direction,
                                                                 **self.mixed_image_to_mixed_isr_parms).cuda()
                        mixed_isr[i] = mixed_isr[i].repeat(3, 1, 1)[None]
                    else:
                        mixed_isr_list = []
                        for j in range(3):
                            mixed_isr_list.append(get_image_change_from_pil(mixed_i_pil, width=512, height=512,
                                                                            **self.mixed_image_to_mixed_isr_parms[j],
                                                                            auto_threshold=None).cuda())
                        mixed_isr_list = torch.cat(mixed_isr_list, dim=0)
                        mixed_isr[i] = mixed_isr_list[None]
                    if self.isr_noise_dacs_type != '':
                        mixed_isr[i] = add_noise_on_isr(mixed_isr[i][0, 0:1], transform_type=self.isr_noise_dacs_type)[None]
                        mixed_isr[i] = mixed_isr[i].repeat(1, 3, 1, 1)
                else:
                    _, mixed_isr[i] = strong_transform(strong_parameters, target=torch.stack((day_isr[i], night_isr[i])),
                                                       isr_flag=self.isr_mix_aug)
            if self.train_type == 'cs2dz_image+raw-isr_split':
                _, pseudo_weight_image[i] = strong_transform(strong_parameters, target=torch.stack((gt_pixel_weight[i], pseudo_weight_image[i])))
                _, pseudo_weight_events[i] = strong_transform(strong_parameters, target=torch.stack((gt_pixel_weight[i], pseudo_weight_events[i])))
                _, mixed_lbl[i] = strong_transform(strong_parameters, target=torch.stack((day_label[i][0], pseudo_label_image[i])))
                _, mixed_lbl_2[i] = strong_transform(strong_parameters, target=torch.stack((day_label[i][0], pseudo_label_events[i])))
            else:
                _, mixed_lbl[i] = strong_transform(strong_parameters, target=torch.stack((day_label[i][0], pseudo_label[i])))
                _, pseudo_weight[i] = strong_transform(strong_parameters, target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
        mixed_img = torch.cat(mixed_img) if mixed_img[0] is not None else None
        mixed_events = torch.cat(mixed_events) if mixed_events[0] is not None else None
        mixed_lbl = torch.cat(mixed_lbl)
        mixed_isr = torch.cat(mixed_isr) if mixed_isr[0] is not None else None
        mixed_lbl_2 = torch.cat(mixed_lbl_2) if self.train_type == 'cs2dz_image+raw-isr_split' else None
        if self.train_type == 'cs2dz_image+d2n-isr':
            if mixed_events is None:
                mixed_events = [None] * batch_size
            mix_masks_target_isr = get_class_masks(target_day_pl)  # 0(target) or 1(source)
            for i in range(batch_size):
                strong_parameters['mix'] = mix_masks_target_isr[i]
                _, mixed_lbl_2[i] = strong_transform(strong_parameters,
                                                     target=torch.stack((target_day_pl[i][0], pseudo_label[i])))
                assert day_events is None
                _, mixed_events[i] = strong_transform(strong_parameters,
                                                      target=torch.stack((target_day_t_isr[i], night_isr[i])))
            mixed_lbl_2 = torch.cat(mixed_lbl_2)
            mixed_events = torch.cat(mixed_events)

        ################################################################
        ################### Train on mixed images
        ################################################################
        if self.train_type in {'cs2dsec_image', 'cs2dz_image'}:
            mix_losses, pred = self.get_model().forward_train(mixed_img, mixed_events, mixed_lbl,
                                                              seg_weight=pseudo_weight, return_feat=True)
        elif self.train_type == 'cs2dz_image+d2n-isr':
            inputs = {'image': mixed_img, 'events': mixed_events}
            target_lbl = {'image': mixed_lbl, 'events': mixed_lbl_2}
            mix_losses, pred = self.get_model().forward_train(inputs, target_lbl, seg_weight=pseudo_weight,
                                                              return_feat=True, cfg=self.forward_cfg)
        elif self.train_type == 'cs2dz_image+raw-isr_split':
            inputs = {'image': mixed_img, 'events': mixed_isr}
            target_lbl = {'image': mixed_lbl, 'events': mixed_lbl_2}
            pseudo_weight_dict = {'image': pseudo_weight_image, 'events': pseudo_weight_events}
            mix_losses, pred = self.get_model().forward_train(inputs, target_lbl, seg_weight=pseudo_weight_dict,
                                                              return_feat=True, cfg=self.forward_cfg)
        elif self.train_type == 'cs2dz_image+raw-isr':
            inputs = {'image': mixed_img, 'events': mixed_isr}
            mix_losses, pred = self.get_model().forward_train(inputs, mixed_lbl, seg_weight=pseudo_weight,
                                                              return_feat=True, cfg=self.forward_cfg)
        elif self.train_type == 'cs2dz_image+raw-isr_no-fusion':
            with torch.no_grad():
                mixed_isr_features = self.get_model().extract_feat(image=None, events=mixed_isr)['f_events']
                self.forward_cfg['mixed_isr_features'] = mixed_isr_features
                self.forward_cfg['lambda_isr_features'] = self.lambda_isr_features
            inputs = {'image': mixed_img, 'events': mixed_isr}
            if self.isr_edge:
                target_lbl = {'image': mixed_lbl, 'events': seg_label_to_edge_label(mixed_lbl)}
            else:
                target_lbl = mixed_lbl
            mix_losses, pred = self.get_model().forward_train(inputs, target_lbl, seg_weight=pseudo_weight,
                                                              return_feat=True, cfg=self.forward_cfg)
            self.forward_cfg['mixed_isr_features'] = None
        elif self.train_type == 'cs2dsec_image+events_together':
            inputs = {'image': mixed_img, 'events': mixed_events, 'img_self_res': mixed_isr}
            if self.fuse_both_ice_and_e:
                mix_losses, pred = self.get_model().forward_train(inputs, mixed_lbl, seg_weight=pseudo_weight,
                                                                  return_feat=True,
                                                                  cfg=dict(self.forward_cfg, **{'fusion_all': True}))
            elif self.isr_another_fusion and not (self.forward_cfg['isr_events_fusion_choice'] > self.random_choice_thres):
                mix_losses, pred = self.get_model().forward_train(inputs, mixed_lbl, seg_weight=pseudo_weight,
                                                                  return_feat=True,
                                                                  cfg=dict(self.forward_cfg, **{'fusion_isr': True}))
            else:
                mix_losses, pred = self.get_model().forward_train(inputs, mixed_lbl, seg_weight=pseudo_weight,
                                                                  return_feat=True, cfg=self.forward_cfg)
        else:
            assert self.train_type == 'cs2dsec_image+events'
            inputs = {'image': mixed_img}
            if self.forward_cfg['isr_events_fusion_choice'] > self.random_choice_thres:  # events
                inputs['events'] = mixed_events
            else:  # isr
                inputs['events'] = mixed_isr
            if self.isr_no_fusion and not (self.forward_cfg['isr_events_fusion_choice'] > self.random_choice_thres):  # isr
                mix_losses, pred = self.get_model().forward_train(inputs, mixed_lbl, seg_weight=pseudo_weight, return_feat=True,
                                                                  cfg=dict(self.forward_cfg, **{'no_fusion': True}))
            elif self.isr_another_fusion and not (self.forward_cfg['isr_events_fusion_choice'] > self.random_choice_thres):
                mix_losses, pred = self.get_model().forward_train(inputs, mixed_lbl, seg_weight=pseudo_weight, return_feat=True,
                                                                  cfg=dict(self.forward_cfg, **{'fusion_isr': True}))
            else:
                mix_losses, pred = self.get_model().forward_train(inputs, mixed_lbl, seg_weight=pseudo_weight,
                                                                  return_feat=True, cfg=self.forward_cfg)

        mix_losses.pop('features')  # dict_keys(['features', 'decode.loss_seg', 'decode.acc_seg'])
        mix_losses = add_prefix(mix_losses, 'mix')  # dict_keys(['mix.decode.loss_seg', 'mix.decode.acc_seg'])
        # mix_loss = tensor(2.5249, device='cuda:0', grad_fn=<AddBackward0>)
        # mix_log_vars = OrderedDict([('mix.decode.loss_seg', 2.524905204772949),
        # ('mix.decode.acc_seg', 39.74323272705078), ('loss', 2.524905204772949)])
        mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        # log_vars = dict_keys(['decode.loss_seg', 'decode.acc_seg', 'loss'])
        log_vars.update(mix_log_vars)
        # log_vars = dict_keys(['decode.loss_seg', 'decode.acc_seg', 'loss', 'mix.decode.loss_seg', 'mix.decode.acc_seg'])
        target_loss = mix_loss
        target_loss.backward()

        ################################################################
        ################### create mix data visualization
        ################################################################
        with torch.no_grad():
            if self.train_type in {'cs2dsec_image', 'cs2dz_image'}:
                mix_img_softmax = torch.softmax(pred, dim=1)  # img
                _, mix_img_seg = torch.max(mix_img_softmax, dim=1)
            else:
                mix_img_softmax = torch.softmax(pred['image_output'], dim=1)  # img
                _, mix_img_seg = torch.max(mix_img_softmax, dim=1)
                mix_events_softmax = torch.softmax(pred['events_output'], dim=1)  # events
                _, mix_events_seg = torch.max(mix_events_softmax, dim=1)
                if self.train_type not in {'cs2dz_image+d2n-isr', 'cs2dz_image+raw-isr_split', 'cs2dz_image+raw-isr_no-fusion'}:
                    if not self.isr_no_fusion or self.forward_cfg['isr_events_fusion_choice'] > self.random_choice_thres:  # events
                        mix_fusion_softmax = torch.softmax(pred['fusion_output'], dim=1)  # fusion
                        _, mix_fusion_seg = torch.max(mix_fusion_softmax, dim=1)
                if self.train_type == 'cs2dsec_image+events_together':
                    mix_isr_softmax = torch.softmax(pred['img_self_res_output'], dim=1)  # fusion
                    _, mix_isr_seg = torch.max(mix_isr_softmax, dim=1)

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'], 'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)

            if day_image is None:
                vis_img = torch.clamp(torch.mean((day_events + 1) / 2, dim=1, keepdim=True).repeat(1, 3, 1, 1), 0, 1)
                vis_trg_img = vis_img
                vis_mixed_img = vis_img
            else:
                vis_img = torch.clamp(denorm(day_image, means, stds), 0, 1)  # [B, 3, H, W] range: 0~1
                vis_trg_img = torch.clamp(denorm(night_image, means, stds), 0, 1)
                vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)

            if day_events is None:
                vis_events = vis_img
                vis_trg_events = vis_img
                vis_mixed_events = vis_img
            else:
                vis_events = torch.clamp(torch.mean((day_events + 1) / 2, dim=1, keepdim=True).repeat(1, 3, 1, 1), 0, 1)
                vis_trg_events = torch.clamp(torch.mean((night_events + 1) / 2, dim=1, keepdim=True).repeat(1, 3, 1, 1), 0, 1)
                vis_mixed_events = torch.clamp(torch.mean((mixed_events + 1) / 2, dim=1, keepdim=True).repeat(1, 3, 1, 1), 0, 1)

            if self.train_type == 'cs2dz_image+d2n-isr':
                vis_mixed_events = torch.clamp(torch.mean((mixed_events + 1) / 2, dim=1, keepdim=True).repeat(1, 3, 1, 1), 0, 1)
                vis_day_isr = torch.clamp((target_day_t_isr + 1) / 2, 0, 1)
                vis_night_isr = torch.clamp((night_isr + 1) / 2, 0, 1)
            elif self.train_type in {'cs2dsec_image+events', 'cs2dsec_image+events_together'}:
                vis_day_isr = torch.clamp((day_isr + 1) / 2, 0, 1)
                vis_night_isr = torch.clamp((night_isr + 1) / 2, 0, 1)
                vis_mixed_isr = torch.clamp((mixed_isr + 1) / 2, 0, 1)
            elif self.train_type in {'cs2dz_image+raw-isr', 'cs2dz_image+raw-isr_split', 'cs2dz_image+raw-isr_no-fusion'}:
                vis_day_isr = torch.clamp((day_isr + 1) / 2, 0, 1)
                vis_night_isr = torch.clamp((night_isr + 1) / 2, 0, 1)
                vis_mixed_isr = torch.clamp((mixed_isr + 1) / 2, 0, 1)

            for j in range(batch_size):
                if self.train_type in {'cs2dsec_image', 'cs2dz_image'}:
                    rows, cols = 2, 5
                else:
                    rows, cols = 4, 6
                fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), gridspec_kw={'hspace': 0.1,
                    'wspace': 0, 'top': 0.95, 'bottom': 0, 'right': 1, 'left': 0},)

                if self.train_type in {'cs2dsec_image', 'cs2dz_image'}:
                    subplotimg(axs[0][0], vis_img[j], 'Source Image')
                    subplotimg(axs[0][1], day_label[j], 'Source GT Seg', cmap='cityscapes')
                    subplotimg(axs[0][2], vis_mixed_img[j], 'Mixed Image')
                    subplotimg(axs[0][3], pseudo_weight[j], 'Pseudo W.', vmin=0, vmax=1)
                    if self.debug_fdist_mask is not None:
                        subplotimg(axs[0][4], self.debug_fdist_mask[j][0], 'FDist Mask', cmap='gray')

                    subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                    subplotimg(axs[1][1], pseudo_label[j], 'Target Fusion Seg', cmap='cityscapes')
                    subplotimg(axs[1][2], mix_masks[j][0], 'Domain Mask', cmap='gray')
                    subplotimg(axs[1][3], mixed_lbl[j], 'Mixed PL Seg', cmap='cityscapes')
                    if self.debug_gt_rescale is not None:
                        subplotimg(axs[1][4], self.debug_gt_rescale[j], 'Scaled GT', cmap='cityscapes')
                elif self.train_type == 'cs2dz_image+d2n-isr':
                    target_day_vis_img = torch.clamp(denorm(target_day_image, means, stds), 0, 1)

                    subplotimg(axs[0][0], vis_img[j], 'Source Image')
                    subplotimg(axs[0][1], vis_day_isr[j], 'Target Day ISR\'')
                    subplotimg(axs[0][2], day_img_seg[j], 'Source Image Seg', cmap='cityscapes')
                    subplotimg(axs[0][3], day_events_seg[j], 'Target Day ISR\' Seg', cmap='cityscapes')
                    subplotimg(axs[0][4], day_label[j], 'Source GT Seg', cmap='cityscapes')
                    subplotimg(axs[0][5], target_day_pl[j], 'Target Day PL(GT) Seg', cmap='cityscapes')

                    subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                    subplotimg(axs[1][1], vis_night_isr[j], 'Target ISR')
                    subplotimg(axs[1][2], ema_img_seg[j], 'Target Image Seg', cmap='cityscapes')
                    subplotimg(axs[1][3], ema_events_seg[j], 'Target ISR Seg', cmap='cityscapes')
                    subplotimg(axs[1][4], pseudo_label[j], 'Target Avg-Fusion Seg', cmap='cityscapes')

                    subplotimg(axs[2][0], vis_mixed_img[j], 'Mixed Image')
                    subplotimg(axs[2][1], vis_mixed_events[j], 'Mixed ISR')
                    subplotimg(axs[2][2], mix_img_seg[j], 'Mixed Image Seg', cmap='cityscapes')
                    subplotimg(axs[2][3], mix_events_seg[j], 'Mixed ISR Seg', cmap='cityscapes')
                    subplotimg(axs[2][4], mixed_lbl[j], 'Mixed Image PL(GT) Seg', cmap='cityscapes')
                    subplotimg(axs[2][5], mixed_lbl_2[j], 'Mixed ISR PL(GT) Seg', cmap='cityscapes')

                    subplotimg(axs[3][0], target_day_vis_img[j], 'Target Day Image')
                    subplotimg(axs[3][1], mix_masks_target_isr[j][0], 'Domain Mask T-ISR', cmap='gray')
                    subplotimg(axs[3][2], mix_masks[j][0], 'Domain Mask S-Image', cmap='gray')
                    if self.debug_fdist_mask is not None:
                        subplotimg(axs[3][3], self.debug_fdist_mask[j][0], 'FDist Mask', cmap='gray')
                    if self.debug_gt_rescale is not None:
                        subplotimg(axs[3][4], self.debug_gt_rescale[j], 'Scaled GT', cmap='cityscapes')
                elif self.train_type in {'cs2dz_image+raw-isr', 'cs2dz_image+raw-isr_split', 'cs2dz_image+raw-isr_no-fusion'}:
                    subplotimg(axs[0][0], vis_img[j], 'Source Image')
                    subplotimg(axs[0][1], vis_day_isr[j], 'Source ISR')
                    subplotimg(axs[0][2], day_img_seg[j], 'Source Image Seg', cmap='cityscapes')
                    subplotimg(axs[0][3], day_events_seg[j], 'Source ISR Seg', cmap='cityscapes')
                    if self.train_type == 'cs2dz_image+raw-isr':
                        subplotimg(axs[0][4], day_fusion_seg[j], 'Source Fusion Seg', cmap='cityscapes')
                    elif self.train_type == 'cs2dz_image+raw-isr_no-fusion' and self.isr_edge:
                        subplotimg(axs[0][4], seg_label_to_edge_label(day_label)[j], 'Source GT Seg', cmap='cityscapes')
                    subplotimg(axs[0][5], day_label[j], 'Source GT Seg', cmap='cityscapes')

                    subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                    subplotimg(axs[1][1], vis_night_isr[j], 'Target ISR')
                    subplotimg(axs[1][2], ema_img_seg[j], 'Target Image Seg', cmap='cityscapes')
                    subplotimg(axs[1][3], ema_events_seg[j], 'Target ISR Seg', cmap='cityscapes')
                    if self.train_type == 'cs2dz_image+raw-isr':
                        subplotimg(axs[1][4], pseudo_label_f[j], 'Target Fusion Seg', cmap='cityscapes')
                    subplotimg(axs[1][5], mix_masks[j][0], 'Domain Mask', cmap='gray')

                    subplotimg(axs[2][0], vis_mixed_img[j], 'Mixed Image')
                    subplotimg(axs[2][1], vis_mixed_isr[j], 'Mixed ISR')
                    subplotimg(axs[2][2], mix_img_seg[j], 'Mixed Image Seg', cmap='cityscapes')
                    subplotimg(axs[2][3], mix_events_seg[j], 'Mixed ISR Seg', cmap='cityscapes')
                    if self.train_type == 'cs2dz_image+raw-isr':
                        subplotimg(axs[2][4], mix_fusion_seg[j], 'Mixed Fusion Seg', cmap='cityscapes')
                    elif self.train_type == 'cs2dz_image+raw-isr_split':
                        subplotimg(axs[2][4], mixed_lbl_2[j], 'Mixed ISR PL(GT) Seg', cmap='cityscapes')
                    elif self.train_type == 'cs2dz_image+raw-isr_no-fusion' and self.isr_edge:
                        subplotimg(axs[2][4], seg_label_to_edge_label(mixed_lbl)[j], 'Source GT Seg', cmap='cityscapes')
                    subplotimg(axs[2][5], mixed_lbl[j], 'Mixed image PL(GT) Seg', cmap='cityscapes')
                    if self.debug_fdist_mask is not None:
                        subplotimg(axs[3][3], self.debug_fdist_mask[j][0], 'FDist Mask', cmap='gray')
                    if self.debug_gt_rescale is not None:
                        subplotimg(axs[3][4], self.debug_gt_rescale[j], 'Scaled GT', cmap='cityscapes')

                    if self.deflare_aug:
                        vis_trg_img_deflare = torch.clamp(denorm(night_image_deflare, means, stds), 0, 1)
                        vis_night_isr_deflare = torch.clamp(torch.mean((night_isr_deflare + 1) / 2, dim=1, keepdim=True).repeat(1, 3, 1, 1), 0, 1)
                        subplotimg(axs[3][0], vis_trg_img_deflare[j], 'Target Image_deflare', cmap='gray')
                        subplotimg(axs[3][1], vis_night_isr_deflare[j], 'Target ISR_deflare', cmap='gray')
                elif self.train_type == 'cs2dsec_image+events_together':

                    if self.forward_cfg['isr_events_fusion_choice'] > self.random_choice_thres:  # events
                        source_caption = 'Source Fusion(I+E) Seg'
                        target_caption = 'Target Fusion(I+E) Seg'
                        mix_caption = 'Mixed Fusion(I+E) Seg'
                    else:
                        source_caption = 'Source Fusion(I+SF) Seg'
                        target_caption = 'Target Fusion(I+SF) Seg'
                        mix_caption = 'Mixed Fusion(I+SF) Seg'

                    subplotimg(axs[0][0], vis_img[j], 'Source Image')
                    subplotimg(axs[0][1], vis_events[j], 'Source Events')
                    subplotimg(axs[0][2], day_img_seg[j], 'Source Image Seg', cmap='cityscapes')
                    subplotimg(axs[0][3], day_events_seg[j], 'Source Events Seg', cmap='cityscapes')
                    subplotimg(axs[0][4], day_fusion_seg[j], source_caption, cmap='cityscapes')
                    subplotimg(axs[0][5], day_label[j], 'Source GT Seg', cmap='cityscapes')

                    subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                    subplotimg(axs[1][1], vis_trg_events[j], 'Target Events')
                    subplotimg(axs[1][2], ema_img_seg[j], 'Target Image Seg', cmap='cityscapes')
                    subplotimg(axs[1][3], ema_events_seg[j], 'Target Events Seg', cmap='cityscapes')
                    subplotimg(axs[1][4], pseudo_label_f[j], target_caption, cmap='cityscapes')
                    subplotimg(axs[1][5], mix_masks[j][0], 'Domain Mask', cmap='gray')

                    subplotimg(axs[2][0], vis_mixed_img[j], 'Mixed Image')
                    subplotimg(axs[2][1], vis_mixed_events[j], 'Mixed Events')
                    subplotimg(axs[2][2], mix_img_seg[j], 'Mixed Image Seg', cmap='cityscapes')
                    subplotimg(axs[2][3], mix_events_seg[j], 'Mixed Events Seg', cmap='cityscapes')
                    subplotimg(axs[2][4], mix_fusion_seg[j], mix_caption, cmap='cityscapes')
                    subplotimg(axs[2][5], mixed_lbl[j], 'Mixed PL Seg (PL)', cmap='cityscapes')

                    subplotimg(axs[3][0], vis_day_isr[j], 'Source img_self_res')
                    subplotimg(axs[3][1], day_isr_seg[j], 'Source img_self_res Seg', cmap='cityscapes')
                    subplotimg(axs[3][2], vis_night_isr[j], 'Target img_self_res')
                    # subplotimg(axs[3][3], ema_isr_seg[j], 'Target img_self_res Seg', cmap='cityscapes')
                    subplotimg(axs[3][4], vis_mixed_isr[j], 'Mixed img_self_res')
                    subplotimg(axs[3][5], mix_isr_seg[j], 'Mixed img_self_res Seg', cmap='cityscapes')
                else:
                    assert self.train_type == 'cs2dsec_image+events'

                    subplotimg(axs[0][0], vis_img[j], 'Source Image')
                    # subplotimg(axs[0][1], vis_events[j], 'Source Events')
                    subplotimg(axs[0][2], day_img_seg[j], 'Source Image Seg', cmap='cityscapes')
                    subplotimg(axs[0][3], day_events_seg[j], 'Source Events Seg', cmap='cityscapes')
                    # subplotimg(axs[0][4], day_fusion_seg[j], 'Source Fusion Seg', cmap='cityscapes')
                    subplotimg(axs[0][5], day_label[j], 'Source GT Seg', cmap='cityscapes')

                    subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                    # subplotimg(axs[1][1], vis_trg_events[j], 'Target Events')
                    subplotimg(axs[1][2], ema_img_seg[j], 'Target Image Seg', cmap='cityscapes')
                    subplotimg(axs[1][3], ema_events_seg[j], 'Target Events Seg', cmap='cityscapes')
                    subplotimg(axs[1][4], pseudo_label_f[j], 'Target Fusion Seg', cmap='cityscapes')
                    subplotimg(axs[1][5], mix_masks[j][0], 'Domain Mask', cmap='gray')

                    subplotimg(axs[2][0], vis_mixed_img[j], 'Mixed Image')
                    # subplotimg(axs[2][1], vis_mixed_events[j], 'Mixed Events')
                    subplotimg(axs[2][2], mix_img_seg[j], 'Mixed Image Seg', cmap='cityscapes')
                    subplotimg(axs[2][3], mix_events_seg[j], 'Mixed Events Seg', cmap='cityscapes')
                    # subplotimg(axs[2][4], mix_fusion_seg[j], 'Mixed Fusion Seg', cmap='cityscapes')
                    subplotimg(axs[2][5], mixed_lbl[j], 'Mixed PL Seg (PL)', cmap='cityscapes')

                    if not self.isr_no_fusion or self.forward_cfg['isr_events_fusion_choice'] > self.random_choice_thres:  # events
                        subplotimg(axs[0][4], day_fusion_seg[j], 'Source Fusion Seg', cmap='cityscapes')
                        subplotimg(axs[2][4], mix_fusion_seg[j], 'Mixed Fusion Seg', cmap='cityscapes')

                    if self.forward_cfg['isr_events_fusion_choice'] > self.random_choice_thres:
                        subplotimg(axs[0][1], vis_events[j], 'Source Events')
                        subplotimg(axs[1][1], vis_trg_events[j], 'Target Events')
                        subplotimg(axs[2][1], vis_mixed_events[j], 'Mixed Events')
                    else:
                        subplotimg(axs[0][1], vis_day_isr[j], 'Source img_self_res')
                        subplotimg(axs[1][1], vis_night_isr[j], 'Target img_self_res')
                        subplotimg(axs[2][1], vis_mixed_isr[j], 'Mixed img_self_res')
                    '''subplotimg(axs[3][0], vis_day_isr[j], 'Source img_self_res')
                    subplotimg(axs[3][1], day_isr_seg[j], 'Source img_self_res Seg', cmap='cityscapes')
                    subplotimg(axs[3][2], vis_night_isr[j], 'Target img_self_res')
                    subplotimg(axs[3][3], ema_isr_seg[j], 'Target img_self_res Seg', cmap='cityscapes')
                    subplotimg(axs[3][4], vis_mixed_isr[j], 'Mixed img_self_res')
                    subplotimg(axs[3][5], mix_isr_seg[j], 'Mixed img_self_res Seg', cmap='cityscapes')'''

                    if self.debug_fdist_mask is not None:
                        subplotimg(axs[3][6], self.debug_fdist_mask[j][0], 'FDist Mask', cmap='gray')
                    if self.debug_gt_rescale is not None:
                        subplotimg(axs[3][7], self.debug_gt_rescale[j], 'Scaled GT', cmap='cityscapes')
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(os.path.join(out_dir, f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()
        self.local_iter += 1
        return log_vars


@UDA.register_module()
class OrgDACS(UDADecorator):

    def __init__(self, **cfg):
        super(OrgDACS, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.debug_img_interval = cfg['debug_img_interval']
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        with torch.no_grad():
            self.get_imnet_model().eval()
            feat_imnet = self.get_imnet_model().extract_feat(img)
            feat_imnet = [f.detach() for f in feat_imnet]
        lay = -1
        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                self.fdist_scale_min_ratio,
                                                self.num_classes,
                                                255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                              fdist_mask)
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescale = gt_rescaled
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        # Train on source images
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features')
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=self.enable_fdist)
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                      src_feat)
            feat_loss.backward()
            log_vars.update(add_prefix(feat_log, 'src'))
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')

        # Generate pseudo-label
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        ema_logits = self.get_ema_model().encode_decode(
            target_img, target_img_metas)

        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=dev)

        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

        # Apply mixing
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mix_masks = get_class_masks(gt_semantic_seg)

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
            _, pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)

        # Train on mixed images
        mix_losses = self.get_model().forward_train(
            mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=True)
        mix_losses.pop('features')
        mix_losses = add_prefix(mix_losses, 'mix')
        mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(mix_log_vars)
        mix_loss.backward()

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'],
                                   'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            for j in range(batch_size):
                rows, cols = 2, 5
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                subplotimg(
                    axs[0][1],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap='cityscapes')
                subplotimg(
                    axs[1][1],
                    pseudo_label[j],
                    'Target Seg (Pseudo) GT',
                    cmap='cityscapes')
                subplotimg(axs[0][2], vis_mixed_img[j], 'Mixed Image')
                subplotimg(
                    axs[1][2], mix_masks[j][0], 'Domain Mask', cmap='gray')
                # subplotimg(axs[0][3], pred_u_s[j], "Seg Pred",
                #            cmap="cityscapes")
                subplotimg(
                    axs[1][3], mixed_lbl[j], 'Seg Targ', cmap='cityscapes')
                subplotimg(
                    axs[0][3], pseudo_weight[j], 'Pseudo W.', vmin=0, vmax=1)
                if self.debug_fdist_mask is not None:
                    subplotimg(
                        axs[0][4],
                        self.debug_fdist_mask[j][0],
                        'FDist Mask',
                        cmap='gray')
                if self.debug_gt_rescale is not None:
                    subplotimg(
                        axs[1][4],
                        self.debug_gt_rescale[j],
                        'Scaled GT',
                        cmap='cityscapes')
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()
        self.local_iter += 1

        return log_vars