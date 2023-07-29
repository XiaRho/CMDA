# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Support for seg_weight

from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from ..builder import build_loss
from ..losses import accuracy


class BaseDecodeHead(BaseModule, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 decoder_params=None,
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='conv_seg'))):
        super(BaseDecodeHead, self).__init__(init_cfg)
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      train_cfg,
                      seg_weight=None):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)
        return losses, seg_logits

    def forward_test(self, inputs, img_metas=None, test_cfg=None):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label, seg_weight=None):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss


class BaseDecodeHeadFusion(BaseModule, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 decoder_params=None,
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 init_cfg=dict(type='Normal', std=0.01, override=dict(name='conv_seg'))):
        super(BaseDecodeHeadFusion, self).__init__(init_cfg)
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)  # 256-->19
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        self.split_cls = False
        assert 'train_type' in decoder_params.keys()
        self.train_type = decoder_params['train_type']
        if self.train_type == 'cs2dz_image+raw-isr_split':
            self.split_cls = True
            self.conv_seg_events = nn.Conv2d(channels, num_classes, kernel_size=1)
            self.dropout_events = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
            self.conv_seg_fusion = nn.Conv2d(channels, num_classes, kernel_size=1)
            self.dropout_fusion = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
        elif self.train_type == 'cs2dz_image+raw-isr_no-fusion':
            self.split_cls = True
            self.conv_seg_events = nn.Conv2d(channels, 2, kernel_size=1)
            self.dropout_events = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None

        if 'share_decoder' in decoder_params and decoder_params['share_decoder']:
            self.share_decoder = True
        else:
            self.share_decoder = False

        if 'half_share_decoder' in decoder_params and decoder_params['half_share_decoder']:
            self.half_share_decoder = True
            assert not self.share_decoder
        else:
            self.half_share_decoder = False

        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      train_cfg,
                      seg_weight=None,
                      cfg=None):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs, cfg)
        # seg_logits['fusion_output']: [2, 19, 128, 128] (-0.7546 ~ 0.6156)
        # gt_semantic_seg: [2, 1, 512, 512] (0 ~ 19 + 255)
        # seg_weight: [2, 512, 512] (0 ~ 1)
        if 'cal_confidence' in cfg.keys() and cfg['cal_confidence']:
            if seg_weight is None:  # source
                seg_weight = torch.ones_like(gt_semantic_seg)[:, 0].cuda()  # [2, 512, 512]
            _, fusion_out = torch.max(seg_logits['fusion_output'], dim=1)  # [2, 128, 128] (0 ~ 18)
            _, image_out = torch.max(seg_logits['image_output'], dim=1)
            _, events_out = torch.max(seg_logits['events_output'], dim=1)

            diff_image_fusion = torch.ne(fusion_out, image_out)
            same_image_fusion = torch.eq(fusion_out, image_out)
            diff_events_fusion = torch.ne(fusion_out, events_out)
            same_events_fusion = torch.eq(fusion_out, events_out)

            less_focus_image_index = (diff_image_fusion * same_events_fusion)[None].float()
            less_focus_events_index = (diff_events_fusion * same_image_fusion)[None].float()

            less_focus_image_index = resize(input=less_focus_image_index, size=seg_weight.shape[1:])[0].bool()
            less_focus_events_index = resize(input=less_focus_events_index, size=seg_weight.shape[1:])[0].bool()

            if cfg['confidence_type'] == 'soft_gradual':
                image_attention = torch.logical_not(less_focus_image_index).float() + \
                                  less_focus_image_index.float() * (1 - cfg['gradual_rate'])
                events_attention = torch.logical_not(less_focus_events_index).float() + \
                                   less_focus_events_index.float() * (1 - cfg['gradual_rate'])
            elif cfg['confidence_type'] == 'hard':
                image_attention = torch.logical_not(less_focus_image_index)
                events_attention = torch.logical_not(less_focus_events_index)
            else:
                raise ValueError('error confidence_type')
            image_seg_weight = seg_weight * image_attention
            events_seg_weight = seg_weight * events_attention
        else:
            if seg_weight is None:
                if isinstance(gt_semantic_seg, dict):
                    seg_weight = torch.ones_like(gt_semantic_seg['image'])[:, 0].cuda()
                else:
                    seg_weight = torch.ones_like(gt_semantic_seg)[:, 0].cuda()
            if isinstance(seg_weight, dict):
                image_seg_weight = seg_weight['image']
                events_seg_weight = seg_weight['events']
            else:
                image_seg_weight = seg_weight
                events_seg_weight = seg_weight

        losses = dict()

        if isinstance(gt_semantic_seg, dict):
            image_gt, events_gt = gt_semantic_seg['image'], gt_semantic_seg['events']
            assert seg_logits['img_self_res_output'] is None
            assert seg_logits['fusion_output'] is None
        else:
            image_gt, events_gt, fusion_gt, isr_gt = gt_semantic_seg, gt_semantic_seg, gt_semantic_seg, gt_semantic_seg

        if self.train_type == 'cs2dz_image+raw-isr_split':
            assert cfg['loss_weight']['image'] == 0.5 and cfg['loss_weight']['events'] == 0.5
            losses_1 = self.losses(seg_logits['image_output'], image_gt, image_seg_weight.detach())
            losses_2 = self.losses(seg_logits['events_output'], events_gt, events_seg_weight.detach())
            losses['loss_seg'] = losses_1['loss_seg'] * cfg['loss_weight']['image'] * 2 + \
                                 losses_2['loss_seg'] * cfg['loss_weight']['events'] * 2
            losses['acc_seg'] = losses_1['acc_seg']
        else:
            losses_2 = self.losses(seg_logits['image_output'], image_gt, image_seg_weight.detach())
            losses_3 = self.losses(seg_logits['events_output'], events_gt, events_seg_weight.detach())
            if seg_logits['fusion_output'] is not None:
                losses_1 = self.losses(seg_logits['fusion_output'], fusion_gt, seg_weight.detach())
            else:
                losses_1 = {'loss_seg': torch.tensor(0).detach()}
            losses['loss_seg'] = losses_1['loss_seg'] * cfg['loss_weight']['fusion'] + \
                                 losses_2['loss_seg'] * cfg['loss_weight']['image']

            if seg_logits['img_self_res_output'] is not None:
                losses_4 = self.losses(seg_logits['img_self_res_output'], isr_gt, events_seg_weight.detach())
                losses['loss_seg'] += (losses_4['loss_seg'] * cfg['loss_weight']['img_self_res'] +
                                       losses_3['loss_seg'] * (cfg['loss_weight']['events'] / 2))
            else:
                losses['loss_seg'] += losses_3['loss_seg'] * cfg['loss_weight']['events']

            if seg_logits['fusion_output'] is not None:
                losses['acc_seg'] = losses_1['acc_seg']
            else:
                losses['acc_seg'] = losses_2['acc_seg']
        '''print('image: {:.3f} * {} = {:.3f}, isr: {:.3f} * {} = {:.3f}'.format(losses_2['loss_seg'].item(),
                                                                              cfg['loss_weight']['image'],
                                                                      losses_2['loss_seg'].item() * cfg['loss_weight']['image'],
                                                                      losses_3['loss_seg'].item(), cfg['loss_weight']['events'],
                                                                      losses_3['loss_seg'].item() * cfg['loss_weight']['events']))'''
        return losses, seg_logits

    def forward_test(self, inputs, output_features=False, test_cfg={'output_type': 'fusion'}):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        if output_features:
            return self.forward(inputs)
        else:
            if test_cfg['output_type'] == 'fusion':
                return self.forward(inputs)['fusion_output']
            elif test_cfg['output_type'] == 'image':
                return self.forward(inputs)['image_output']
            elif test_cfg['output_type'] == 'events':
                return self.forward(inputs)['events_output']
            else:
                raise ValueError('error output_type = {}'.format(test_cfg['output_type']))

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def cls_seg_events(self, feat):
        """Classify each pixel."""
        if not self.split_cls:
            return self.conv_seg(feat)
        if self.dropout_events is not None:
            feat = self.dropout_events(feat)
        output = self.conv_seg_events(feat)
        return output

    def cls_seg_fusion(self, feat):
        """Classify each pixel."""
        if not self.split_cls:
            return self.conv_seg(feat)
        if self.dropout_fusion is not None:
            feat = self.dropout_fusion(feat)
        output = self.conv_seg_fusion(feat)
        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label, seg_weight=None):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss