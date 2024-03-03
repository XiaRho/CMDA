# The ema model and the domain-mixing are based on:
# https://github.com/vikolss/DACS

import math
import os
import random
from copy import deepcopy

import mmcv.runner.hooks.logger.text
import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.cyclegan import define_G
from mmseg.models.uda.uda_decorator import UDADecoratorEvents, UDADecoratorFusion, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)
from mmseg.models.utils.visualization import subplotimg
from mmseg.utils.utils import downscale_label_ratio
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
class DACSImage(UDADecoratorFusion):

    def __init__(self, **cfg):
        super(DACSImage, self).__init__(**cfg)
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
        self.transfer_direction = cfg['transfer_direction']
        assert self.transfer_direction in {'isrd2isrn', 'isrn2isrd'}
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

        self.input_modality = cfg['input_modality']
        assert self.input_modality in ['image', 'events', 'image+events']

        self.contrast_config = cfg['contrast_config']
        self.contrast_warmup_iters = cfg['contrast_config']['warmup_iters']
        assert self.contrast_config['target_contract_type'] in ['mix', 'target']

        self.forward_cfg = cfg['forward_cfg']

        self.cyclegan_transfer = define_G().cuda()
        cyclegan_model_pth = torch.load(cfg['cyclegan_transfer_path'])
        self.cyclegan_transfer.load_state_dict(cyclegan_model_pth)
        self.cyclegan_transfer.eval()

        if self.enable_fdist:
            cfg_imnet = deepcopy(cfg['model'])
            if self.input_modality == 'events':
                cfg_imnet['backbone']['in_chans'] = 3
            elif self.input_modality == 'image+events':
                cfg_imnet['type'] = 'EventsEncoderDecoder'
                cfg_imnet['backbone']['type'] = cfg_imnet['backbone_image']['type']
            self.imnet_model = build_segmentor(cfg_imnet)
        else:
            self.imnet_model = None

        self.source_contrast = ContrastCELoss(self.contrast_config)
        self.target_contrast = ContrastCELoss(self.contrast_config)

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
        # outputs = dict(log_vars=log_vars, num_samples=len(data_batch['img_metas']))
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
            feat_imnet = self.get_imnet_model().extract_feat(img, None)
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

        if self.local_iter == 0 and self.transfer_direction == 'isrn2isrd':
            self.get_model().transfer_direction = 'isrn2isrd'
            self.get_model().cyclegan_transfer = self.cyclegan_transfer

        day_label = kwargs['source']['label']
        day_image = kwargs['source']['image']
        if self.transfer_direction == 'isrd2isrn':
            with torch.no_grad():
                day_img_self_res = self.cyclegan_transfer(kwargs['source']['img_self_res'])
        else:
            day_img_self_res = kwargs['source']['img_self_res']

        log_vars = {}
        batch_size = day_image.shape[0]
        dev = day_image.device

        means, stds = get_mean_std(None, dev)
        # Train on source images
        source_ce_losses, pred = self.get_model().forward_train(day_img_self_res, None, day_label, return_feat=True)
        with torch.no_grad():
            day_img_self_res_softmax = torch.softmax(pred, dim=1)  # img_self_res
            _, day_img_self_res_seg = torch.max(day_img_self_res_softmax, dim=1)

        source_ce_loss, clean_log_vars = self._parse_losses(source_ce_losses)  # ['decode.loss_seg', 'decode.acc_seg']
        log_vars.update(clean_log_vars)
        source_ce_loss.backward(retain_graph=self.enable_fdist)

        if (self.local_iter + 1) % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'], 'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)

            vis_img = torch.clamp(denorm(day_image, means, stds), 0, 1)  # [B, 3, H, W] range: 0~1
            vis_day_img_self_res = torch.clamp(torch.mean((day_img_self_res + 1) / 2, dim=1, keepdim=True).repeat(1, 3, 1, 1), 0, 1)

            for j in range(batch_size):
                rows, cols = 1, 4
                fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), gridspec_kw={'hspace': 0.1,
                                                                                               'wspace': 0,
                                                                                               'top': 0.95,
                                                                                               'bottom': 0,
                                                                                               'right': 1,
                                                                                               'left': 0},)
                subplotimg(axs[0], vis_img[j], 'Source Image')
                subplotimg(axs[1], vis_day_img_self_res[j], 'Source Image-Self_Res')
                subplotimg(axs[2], day_img_self_res_seg[j], 'Source Image-Self_Res Seg', cmap='cityscapes')
                subplotimg(axs[3], day_label[j], 'Source GT Seg', cmap='cityscapes')

                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(os.path.join(out_dir, f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()
        self.local_iter += 1
        return log_vars
